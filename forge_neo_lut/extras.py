"""Bracket the Extras runner so every operation sees ungraded input pixels."""
from functools import wraps
from pathlib import Path

from PIL import Image

from .engine import annotate, grade
from .provenance import SOURCE_KEY, SOURCE_PARAM

EXTRAS_NAME = 'LUT original-first upscale'


def mark_source(image, token):
    image._forge_lut_source = token
    image.info[SOURCE_KEY] = token
    return image


def source_parameters(text, token):
    text = text or ''
    start = text.rfind('\n') + 1
    match = SOURCE_PARAM.search(text[start:])
    if match:
        a, b = match.span(1)
        return text[:start+a] + token + text[start+b:]
    return text.rstrip() + (', ' if text.strip() else '') + f'LUT source: {token}'


def before_saved(params, store):
    token = getattr(params.image, '_forge_lut_source', None)
    if not token:
        return
    store.record(token)
    params.pnginfo[SOURCE_KEY] = token
    params.pnginfo['parameters'] = source_parameters(params.pnginfo.get('parameters'), token)
    # Forge's outer Extras function resets image.info; restore it for the gallery.
    params.image.info.update(params.pnginfo)


def after_saved(params, store):
    token = getattr(params.image, '_forge_lut_source', None)
    if token:
        # Include actual encoded JPEG/WebP pixels, not only pre-encoding pixels.
        path = Path(getattr(params.image, 'already_saved_as', params.filename))
        with Image.open(path) as saved:
            store.add_pixels(token, saved)


def _enabled(runner, args):
    for script in runner.scripts_in_preferred_order():
        if script.name == EXTRAS_NAME:
            values = dict(zip(script.controls, args[script.args_from:script.args_to]))
            value = values.get('enabled')
            return True if value is None else bool(value)
    return False


def install_runner(runner, store, lut_directory, read_parameters, state, notify):
    """Idempotent, instance-only wrapper; no global classes or core files change."""
    runner._forge_lut_context = (store, lut_directory, read_parameters, state, notify)
    if hasattr(runner, '_forge_lut_original_run'):
        return
    runner._forge_lut_original_run = runner.run

    @wraps(runner.run)
    def run(pp, args):
        store, lut_directory, read_parameters, state, notify = runner._forge_lut_context
        if not _enabled(runner, args):
            return runner._forge_lut_original_run(pp, args)
        try:
            token = store.find(pp.image, read_parameters(pp.image))
            if token is None:
                return runner._forge_lut_original_run(pp, args)
            original, lut, strength = store.resolve(token, lut_directory)
        except (ValueError, OSError, UnicodeError, KeyError) as exc:
            state.skipped = True
            notify(f'LUT Extras skipped this image: {exc}')
            return
        pp.image = original
        runner._forge_lut_original_run(pp, args)
        if state.skipped:
            return
        try:
            outputs = []
            for output in [pp, *pp.extra_images]:
                raw = output.image
                graded = annotate(grade(raw, lut, strength), lut, strength)
                new_token = store.capture(raw, lut, strength, graded)
                outputs.append((output, mark_source(graded, new_token), new_token))
            for output, graded, new_token in outputs:
                output.image = graded
                output.info.update({'LUT file': lut.name, 'LUT SHA256': lut.sha256,
                                    'LUT strength': strength, 'LUT source': new_token,
                                    'LUT applied after Extras': True})
        except (ValueError, OSError, UnicodeError, KeyError) as exc:
            state.skipped = True
            notify(f'LUT Extras could not finish this image: {exc}')
    runner.run = run
