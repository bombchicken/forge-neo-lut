"""Local trusted manifests link graded pixels to lossless originals, never undo LUTs."""
import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path

from PIL import Image, PngImagePlugin

from .engine import load_lut, validate_strength

SOURCE_KEY = 'Forge LUT source'
TOKEN = re.compile(r'^[0-9a-f]{32}$')
SOURCE_PARAM = re.compile(r'(?:^|,\s*)LUT source:\s*([0-9a-f]{32})(?=,|$)')


def pixel_hash(image):
    rgba = image.convert('RGBA')
    digest = hashlib.sha256(f'{rgba.width}x{rgba.height}:RGBA:'.encode())
    for top in range(0, rgba.height, 128):
        digest.update(rgba.crop((0, top, rgba.width, min(top+128, rgba.height))).tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(data), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SourceStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.RLock()

    def _manifest(self, token):
        if not isinstance(token, str) or not TOKEN.fullmatch(token):
            raise ValueError('Invalid LUT original link. Image metadata cannot specify a file path.')
        return self.directory / 'records' / f'{token}.json'

    def record(self, token):
        try:
            return json.loads(self._manifest(token).read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError('The saved LUT original link is missing or unreadable.') from exc

    def register(self, original_path, lut, strength, graded):
        token = uuid.uuid4().hex
        record = {'original': str(Path(original_path).resolve()) if original_path else None,
                  'original_sha256': file_hash(original_path) if original_path else None, 'lut': lut.name,
                  'lut_sha256': lut.sha256, 'strength': validate_strength(strength), 'pixels': []}
        with self.lock:
            _write_json(self._manifest(token), record)
            self.add_pixels(token, graded)
        return token

    def attach_original(self, token, original_path):
        with self.lock:
            record = self.record(token)
            record['original'] = str(Path(original_path).resolve())
            record['original_sha256'] = file_hash(original_path)
            _write_json(self._manifest(token), record)

    def add_pixels(self, token, image):
        digest = pixel_hash(image)
        with self.lock:
            record = self.record(token)
            if digest not in record['pixels']:
                record['pixels'].append(digest)
                _write_json(self._manifest(token), record)
            path = self.directory / 'pixels' / f'{digest}.json'
            tokens = json.loads(path.read_text()) if path.exists() else []
            if token not in tokens:
                tokens.append(token)
                _write_json(path, tokens)

    def find(self, image, parameters=''):
        line = (parameters or image.info.get('parameters', '')).splitlines()[-1:]
        line = line[0] if line else ''
        match = SOURCE_PARAM.search(line)
        token = getattr(image, '_forge_lut_source', None) or image.info.get(SOURCE_KEY) or (match.group(1) if match else None)
        digest = pixel_hash(image)
        path = self.directory / 'pixels' / f'{digest}.json'
        if token:
            record = self.record(token)
            if digest in record['pixels']:
                return token
            # Extras restores pre-run metadata after returning, including when
            # saving is disabled. Only an exact, unique known pixel match may
            # supersede that stale token; arbitrary edits still fail below.
            if path.exists():
                candidates = json.loads(path.read_text())
                if len(candidates) == 1:
                    return candidates[0]
            raise ValueError('Image pixels have changed since LUT grading; cannot safely select its original.')
        if path.exists():
            candidates = json.loads(path.read_text())
            if len(candidates) != 1:
                raise ValueError('LUT original match is ambiguous; reload the saved graded PNG with metadata.')
            return candidates[0]
        if 'Forge LUT' in image.info or re.search(r'(?:^|,\s*)LUT (?:file|SHA256):', line):
            raise ValueError('This graded image has no saved original link. Use a new linked generation, or upscale its ungraded original manually.')
        return None

    def resolve(self, token, lut_directory):
        record = self.record(token)
        if not record.get('original'):
            raise ValueError('No saved ungraded original is available for this image. Enable original saving for new generations.')
        try:
            if file_hash(record['original']) != record['original_sha256']:
                raise ValueError('The saved ungraded original has changed.')
            lut_directory = Path(lut_directory).resolve()
            lut_path = (lut_directory / record['lut']).resolve()
            if lut_path.parent != lut_directory:
                raise ValueError('Invalid LUT filename in local original record.')
            lut = load_lut(lut_path)
            if lut.sha256 != record['lut_sha256']:
                raise ValueError('The LUT file has changed; restore the recorded LUT before upscaling.')
            with Image.open(record['original']) as image:
                original = image.copy()
            return original, lut, validate_strength(record['strength'])
        except (OSError, KeyError) as exc:
            raise ValueError('The saved original or its LUT is missing or unreadable.') from exc

    def capture(self, original, lut, strength, graded):
        folder = self.directory / 'originals'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{uuid.uuid4().hex}.png'
        metadata = PngImagePlugin.PngInfo()
        for key, value in original.info.items():
            if isinstance(key, str) and isinstance(value, str) and key not in (SOURCE_KEY, 'Forge LUT'):
                metadata.add_text(key, value)
        options = {key: original.info[key] for key in ('icc_profile', 'exif', 'dpi') if key in original.info}
        original.save(path, format='PNG', pnginfo=metadata, **options)
        return self.register(path, lut, strength, graded)


STORE = SourceStore(Path(__file__).resolve().parents[1] / 'data')
