"""Host-independent .cube parsing, grading, and metadata-preserving PNG export."""
import hashlib
import json
import math
import tempfile
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin


@dataclass(frozen=True)
class LUT:
    name: str
    sha256: str
    table: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


@lru_cache(maxsize=4)
def _parse(data: bytes):
    size = None
    minimum = np.zeros(3)
    maximum = np.ones(3)
    seen = set()
    rows = []
    for number, raw in enumerate(data.decode('utf-8-sig').splitlines(), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()
        if key == 'TITLE':
            continue
        if key.startswith('LUT_1D'):
            raise ValueError('Only pure 3D .cube LUTs are supported; 1D/combined LUT found.')
        if key in ('LUT_3D_SIZE', 'DOMAIN_MIN', 'DOMAIN_MAX'):
            if key in seen:
                raise ValueError(f'Duplicate {key} on line {number}.')
            seen.add(key)
            if key == 'LUT_3D_SIZE':
                if len(parts) != 2:
                    raise ValueError('LUT_3D_SIZE needs one integer.')
                size = int(parts[1])
                if not 2 <= size <= 128:
                    raise ValueError('LUT_3D_SIZE must be between 2 and 128.')
            else:
                if len(parts) != 4:
                    raise ValueError(f'{key} needs three numbers.')
                values = np.array(parts[1:], dtype=np.float64)
                if key == 'DOMAIN_MIN':
                    minimum = values
                else:
                    maximum = values
            continue
        if size is None:
            raise ValueError(f'Expected LUT_3D_SIZE before data on line {number}.')
        try:
            if len(parts) != 3:
                raise ValueError()
            rows.append(tuple(float(v) for v in parts))
        except ValueError as exc:
            raise ValueError(f'Invalid LUT data or unsupported directive on line {number}.') from exc
        if len(rows) > size ** 3:
            raise ValueError('Too many LUT data rows.')
    if size is None or len(rows) != size ** 3:
        raise ValueError('LUT data row count must equal LUT_3D_SIZE cubed.')
    table = np.array(rows, dtype=np.float64).reshape(size, size, size, 3)
    if not all(np.isfinite(v).all() for v in (table, minimum, maximum)):
        raise ValueError('LUT contains nonfinite numbers.')
    if not (maximum > minimum).all():
        raise ValueError('Each DOMAIN_MAX must be greater than DOMAIN_MIN.')
    for value in (table, minimum, maximum):
        value.setflags(write=False)
    return table, minimum, maximum


def load_lut(path) -> LUT:
    path = Path(path)
    if path.suffix.lower() != '.cube':
        raise ValueError('Select a .cube file.')
    with path.open('rb') as handle:
        data = handle.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024:
        raise ValueError('LUT file exceeds the 64 MiB limit.')
    table, minimum, maximum = _parse(data)
    return LUT(path.name, hashlib.sha256(data).hexdigest(), table, minimum, maximum)


def validate_strength(strength) -> float:
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('LUT strength must be between 0 and 1.')
    return strength


def grade(image: Image.Image, lut: LUT, strength: float) -> Image.Image:
    strength = validate_strength(strength)
    if strength == 0:
        return image.copy()
    has_alpha = 'A' in image.getbands() or 'transparency' in image.info
    source = image.convert('RGBA' if has_alpha else 'RGB')
    result = source.copy()
    size = lut.table.shape[0]
    # Red is the fastest-changing index in .cube, i.e. table[b, g, r].
    strip_height = max(1, min(128, 65536 // source.width))
    for top in range(0, source.height, strip_height):
        box = (0, top, source.width, min(top + strip_height, source.height))
        pixels = np.array(source.crop(box))
        rgb = pixels[..., :3].astype(np.float64) / 255
        coords = np.clip((rgb - lut.minimum) / (lut.maximum - lut.minimum), 0, 1) * (size - 1)
        low = np.floor(coords).astype(np.intp)
        high = np.minimum(low + 1, size - 1)
        weight = coords - low
        mapped = np.zeros_like(rgb)
        for r in (0, 1):
            for g in (0, 1):
                for b in (0, 1):
                    ri = (low, high)[r][..., 0]
                    gi = (low, high)[g][..., 1]
                    bi = (low, high)[b][..., 2]
                    w = ((weight[..., 0] if r else 1-weight[..., 0]) *
                         (weight[..., 1] if g else 1-weight[..., 1]) *
                         (weight[..., 2] if b else 1-weight[..., 2]))
                    mapped += lut.table[bi, gi, ri] * w[..., None]
        blended = rgb * (1-strength) + mapped * strength
        pixels[..., :3] = np.floor(np.clip(blended, 0, 1)*255 + .5).astype(np.uint8)
        result.paste(Image.fromarray(pixels), box)
    result.info = image.info.copy()
    return result


def annotate(image: Image.Image, lut: LUT, strength: float) -> Image.Image:
    result = image.copy()
    result.info['Forge LUT'] = json.dumps({
        'file': lut.name, 'sha256': lut.sha256,
        'strength': validate_strength(strength), 'interpolation': 'trilinear',
    }, ensure_ascii=False)
    return result


_preview_directory = tempfile.TemporaryDirectory(prefix='forge-neo-lut-')


def save_preview(image: Image.Image) -> str:
    """Unique PNG copy, owned by this process and removed when it exits."""
    output = Path(_preview_directory.name) / f'graded-{uuid.uuid4().hex}.png'
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            pnginfo.add_text(key, value)
    options = {key: image.info[key] for key in ('icc_profile', 'exif', 'dpi') if key in image.info}
    image.save(output, format='PNG', pnginfo=pnginfo, **options)
    return str(output)
