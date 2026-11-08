"""Extras control and runner registration for original-first LUT processing."""
import sys
from pathlib import Path

import gradio as gr
from modules import images, scripts, scripts_postprocessing, shared

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge_neo_lut.extras import EXTRAS_NAME, install_runner
from forge_neo_lut.provenance import STORE


class LUTExtras(scripts_postprocessing.ScriptPostprocessing):
    name = EXTRAS_NAME
    order = 10000

    def __init__(self):
        # Forge creates the runner after importing scripts, then instantiates
        # postprocessing classes. Import-time installation would target None
        # (or the old runner during a reload).
        install_runner(scripts.scripts_postproc, STORE, ROOT / 'luts',
                       lambda image: images.read_info_from_image(image)[0] or '', shared.state, gr.Warning)

    def ui(self):
        return {'enabled': gr.Checkbox(
            label='Upscale ungraded original, then reapply its LUT', value=True,
            info='For linked images made by this extension. Missing originals or changed LUTs skip the image.')}

