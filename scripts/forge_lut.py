"""Forge Neo automatic grading and a generation-free LUT preview tab."""
import sys
import uuid
from copy import copy
from pathlib import Path

import gradio as gr
from modules import images, processing, script_callbacks, scripts
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge_neo_lut.engine import (
    annotate,
    grade,
    load_lut,
    save_preview,
    validate_strength,
)
from forge_neo_lut.extras import after_saved, before_saved, mark_source
from forge_neo_lut.provenance import STORE

LUT_DIR = ROOT / 'luts'
PARAM_KEYS = ('LUT file', 'LUT SHA256', 'LUT strength', 'LUT source')


def lut_names():
    return sorted((p.name for p in LUT_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() == '.cube'), key=str.casefold)


def selected_lut(name):
    if not name or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Choose a LUT from the dropdown. Put .cube files in the luts folder and refresh.')
    path = (LUT_DIR / name).resolve()
    if path.parent != LUT_DIR.resolve():
        raise ValueError('LUT must be inside the extension luts folder.')
    return load_lut(path)


def refresh_luts(current):
    names = lut_names()
    return gr.update(choices=names, value=current if current in names else None)


def lut_controls():
    with gr.Row():
        dropdown = gr.Dropdown(choices=lut_names(), label='3D .cube LUT', value=None, scale=4)
        refresh = gr.Button('Refresh LUTs', scale=1)
    strength = gr.Slider(0, 1, value=1, step=.01, label='LUT strength')
    refresh.click(refresh_luts, inputs=[dropdown], outputs=[dropdown])
    return dropdown, strength


class Script(scripts.Script):
    def title(self):
        return 'LUT Color Grading'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('LUT Color Grading', open=False):
            enabled = gr.Checkbox(label='Enable automatic LUT grading', value=False)
            dropdown, strength = lut_controls()
            original = gr.Checkbox(label='Save an ungraded original too', value=True)
            gr.Markdown('Add `.cube` files to `extensions/forge-neo-lut/luts/`, then refresh. '
                        'Use **LUT Preview** to try a LUT before generating. '
                        'Lossless PNG originals follow your image-saving settings and let Extras '
                        'upscale before applying the LUT.')
        self.infotext_fields = [(enabled, 'LUT enabled'), (dropdown, 'LUT file'), (strength, 'LUT strength')]
        return [enabled, dropdown, strength, original]

    def process(self, p, enabled, name, strength, keep_original):
        # State belongs to this generation request, never the shared Script instance.
        p._forge_neo_lut = None
        p._forge_neo_lut_sources = {}
        for key in (*PARAM_KEYS, 'LUT enabled'):
            p.extra_generation_params.pop(key, None)
        if not enabled:
            return
        try:
            lut = selected_lut(name)
            strength = validate_strength(strength)
        except (ValueError, OSError, UnicodeError) as exc:
            gr.Warning(f'LUT grading disabled for this job: {exc}')
            raise gr.Error(f'Cannot apply LUT: {exc}') from exc
        if strength == 0:
            return
        p._forge_neo_lut = (lut, strength, bool(keep_original))
        def source_parameter(index=0, all_prompts=None, **kwargs):
            # Forge uses batch-local indices in save infotext, global indices
            # in Processed.infotext. Preserve the individual image's ID in both.
            if all_prompts is not None and all_prompts is getattr(p, 'all_prompts', None):
                absolute = index
            else:
                absolute = index + getattr(p, 'iteration', 0) * getattr(p, 'batch_size', 1)
            return p._forge_neo_lut_sources.get(absolute)
        p.extra_generation_params.update({
            'LUT enabled': True, 'LUT file': lut.name,
            'LUT SHA256': lut.sha256, 'LUT strength': strength,
            'LUT source': source_parameter,
        })

    def postprocess_image_after_composite(self, p, pp, *args):
        config = getattr(p, '_forge_neo_lut', None)
        if config is None:
            return
        lut, strength, keep_original = config
        # Calculate first: a grading error cannot replace the source pixels.
        original = pp.image
        pp.image = annotate(grade(original, lut, strength), lut, strength)
        # Recognize grades even if originals are disabled or an original save
        # fails. Extras must skip those, including after metadata-free transfer.
        token = STORE.register(None, lut, strength, pp.image)
        p._forge_neo_lut_sources[pp.index] = token
        mark_source(pp.image, token)
        if keep_original and p.save_samples():
            ungraded_p = copy(p)
            ungraded_p.extra_generation_params = {
                k: v for k, v in p.extra_generation_params.items()
                if k not in (*PARAM_KEYS, 'LUT enabled')
            }
            i = p.batch_index
            info = processing.create_infotext(ungraded_p, p.prompts, p.seeds, p.subseeds,
                                              index=i, all_negative_prompts=p.negative_prompts)
            saved = images.save_image(original.copy(), p.outpath_samples, '', seed=p.seeds[i],
                              prompt=p.prompts[i], extension='png',
                              info=info, p=ungraded_p,
                              suffix=f'-ungraded-{uuid.uuid4().hex[:8]}')
            if saved and saved[0]:
                STORE.attach_original(token, saved[0])


def preview(source_path, name, strength):
    if not source_path:
        return None, None, 'Load an original image, then click Update preview.', None
    try:
        lut = selected_lut(name)
        strength = validate_strength(strength)
        with Image.open(source_path) as opened:
            if opened.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
                raise ValueError('Preview supports 8-bit RGB, RGBA, grayscale and palette images.')
            original = ImageOps.exif_transpose(opened)
            original.load()
        result = annotate(grade(original, lut, strength), lut, strength)
        return original, result, f'Applied {lut.name} at {strength:.0%}. SHA256: {lut.sha256}', result
    except (ValueError, OSError, UnicodeError, Image.DecompressionBombError) as exc:
        return None, None, f'Could not preview: {exc}', None


def export_preview(result):
    if result is None:
        raise gr.Error('Update the preview before saving a graded copy.')
    return save_preview(result)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tab:
        gr.Markdown('## LUT Preview\nLoad an **ungraded original** to compare looks without generating again. '
                    'Each update starts from this file. Your original is never overwritten.')
        source = gr.File(label='Original image', file_types=['image'], type='filepath')
        dropdown, strength = lut_controls()
        update = gr.Button('Update preview', variant='primary')
        with gr.Row():
            original = gr.Image(label='Original', interactive=False, show_download_button=False)
            graded = gr.Image(label='Graded preview', interactive=False, show_download_button=False)
        status = gr.Textbox(label='Status', interactive=False)
        state = gr.State(None)
        save = gr.Button('Save graded PNG copy')
        download = gr.File(label='Download graded copy (includes metadata)', interactive=False)
        update.click(lambda *args: (*preview(*args), None),
                     inputs=[source, dropdown, strength], outputs=[original, graded, status, state, download])
        save.click(export_preview, inputs=[state], outputs=[download])
        source.change(lambda: (None, None, 'Click Update preview.', None, None),
                      outputs=[original, graded, status, state, download])
        for control in (dropdown, strength):
            control.change(lambda: (None, 'Settings changed. Click Update preview.', None, None),
                           outputs=[graded, status, state, download])
    return [(tab, 'LUT Preview', 'forge_neo_lut_preview')]


def before_image_saved(params):
    before_saved(params, STORE)


def image_saved(params):
    try:
        after_saved(params, STORE)
    except (ValueError, OSError) as exc:
        gr.Warning(f'Could not index this LUT image for metadata-free Extras transfer: {exc}')


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_before_image_saved(before_image_saved)
script_callbacks.on_image_saved(image_saved)
