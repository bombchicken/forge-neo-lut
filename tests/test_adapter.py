import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import gradio as gr
from PIL import Image
from test_engine import cube

from forge_neo_lut.provenance import SourceStore


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Forge itself needs model/backend dependencies. Only that host boundary is
        # stubbed; the real Gradio components, image operations and disk I/O run.
        modules = ModuleType('modules')
        modules.scripts = SimpleNamespace(Script=object, AlwaysVisible=object())
        modules.script_callbacks = SimpleNamespace(on_ui_tabs=lambda fn: None,
            on_before_image_saved=lambda fn: None, on_image_saved=lambda fn: None)
        modules.shared = SimpleNamespace(opts=SimpleNamespace(samples_format='png'))
        def create_infotext(p, prompts, seeds, subseeds, index=0, all_negative_prompts=None):
            negatives = p.all_negative_prompts if all_negative_prompts is None else all_negative_prompts
            values = {k: v(index=index, p=p) if callable(v) else v for k, v in p.extra_generation_params.items()}
            return (f'{prompts[index]}\nNegative prompt: {negatives[index]}\nSeed: {seeds[index]}'
                    + ''.join(f', {k}: {v}' for k, v in values.items() if v is not None))
        modules.processing = SimpleNamespace(create_infotext=create_infotext)
        modules.images = SimpleNamespace(save_image=lambda *a, **k: None)
        spec = importlib.util.spec_from_file_location('lut_adapter', Path(__file__).parents[1] / 'scripts' / 'forge_lut.py')
        cls.adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'modules': modules}):
            spec.loader.exec_module(cls.adapter)
        cls.host = modules

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = cube(self.root / 'invert.cube', lambda r, g, b: (1-r, 1-g, 1-b))
        self.directory_patch = patch.object(self.adapter, 'LUT_DIR', self.root)
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)
        self.store = SourceStore(self.root / 'data')
        self.store_patch = patch.object(self.adapter, 'STORE', self.store)
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)

    def request(self):
        return SimpleNamespace(extra_generation_params={}, batch_index=0,
            prompts=['prompt'], seeds=[123], subseeds=[0],
            negative_prompts=['negative'], all_negative_prompts=['negative'],
            outpath_samples=str(self.root), save_samples=lambda: True)

    def test_disabled_and_per_request_isolation(self):
        script = self.adapter.Script()
        p = self.request()
        script.process(p, True, self.path.name, 1, False)
        q = self.request()
        script.process(q, False, '', 1, False)
        pp = SimpleNamespace(image=Image.new('RGB', (1, 1)), index=0)
        script.postprocess_image_after_composite(q, pp)
        self.assertEqual(pp.image.getpixel((0, 0)), (0, 0, 0))
        script.postprocess_image_after_composite(p, pp)
        self.assertEqual(pp.image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(p.extra_generation_params['LUT file'], 'invert.cube')
        self.assertEqual(len(p.extra_generation_params['LUT SHA256']), 64)

    def test_original_save_has_ungraded_pixels_and_metadata(self):
        script = self.adapter.Script()
        p = self.request()
        script.process(p, True, self.path.name, .5, True)
        saved = []
        def save(image, path, basename, **kwargs):
            saved.append((image.copy(), kwargs))
        with patch.object(self.host.images, 'save_image', save):
            pp = SimpleNamespace(image=Image.new('RGB', (1, 1), (0, 100, 255)), index=0)
            script.postprocess_image_after_composite(p, pp)
        self.assertEqual(saved[0][0].getpixel((0, 0)), (0, 100, 255))
        self.assertNotIn('LUT file', saved[0][1]['info'])
        self.assertIn('Seed: 123', saved[0][1]['info'])
        self.assertIn('ungraded', saved[0][1]['suffix'])
        self.assertEqual(pp.image.getpixel((0, 0)), (128, 128, 128))
        self.assertIn('LUT file', p.extra_generation_params)

    def test_later_batch_original_has_current_negative_prompt(self):
        p = self.request()
        p.negative_prompts = ['second negative']
        p.all_negative_prompts = ['first negative', 'second negative']
        script = self.adapter.Script()
        script.process(p, True, self.path.name, 1, True)
        saved = []
        with patch.object(self.host.images, 'save_image', lambda *args, **kw: saved.append(kw)):
            script.postprocess_image_after_composite(p, SimpleNamespace(image=Image.new('RGB', (1, 1)), index=1))
        self.assertIn('Negative prompt: second negative', saved[0]['info'])

    def test_original_save_failure_does_not_discard_grade(self):
        p = self.request()
        script = self.adapter.Script()
        script.process(p, True, self.path.name, 1, True)
        pp = SimpleNamespace(image=Image.new('RGB', (1, 1)), index=0)
        with patch.object(self.host.images, 'save_image', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                script.postprocess_image_after_composite(p, pp)
        self.assertEqual(pp.image.getpixel((0, 0)), (255, 255, 255))

    def test_invalid_configuration_and_traversal_rejected(self):
        for name in ('missing.cube', '../outside.cube', ''):
            with self.subTest(name=name), self.assertRaises(gr.Error):
                self.adapter.Script().process(self.request(), True, name, 1, False)

    def test_preview_reuses_source_and_download_keeps_metadata(self):
        source = self.root / 'source.png'
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        info.add_text('parameters', 'prompt\nSeed: 42')
        Image.new('RGB', (1, 1), (0, 100, 255)).save(source, pnginfo=info)
        a = self.adapter.preview(str(source), self.path.name, 1)
        b = self.adapter.preview(str(source), self.path.name, .5)
        self.assertEqual(a[1].getpixel((0, 0)), (255, 155, 0))
        self.assertEqual(b[1].getpixel((0, 0)), (128, 128, 128))
        with Image.open(source) as original:
            self.assertEqual(original.getpixel((0, 0)), (0, 100, 255))
        path = self.adapter.export_preview(b[3])
        with Image.open(path) as saved:
            self.assertEqual(saved.info['parameters'], 'prompt\nSeed: 42')
            self.assertEqual(json.loads(saved.info['Forge LUT'])['strength'], .5)
        self.assertEqual(self.adapter.preview(None, self.path.name, 1)[3], None)

    def test_actual_gradio_ui_builds_with_no_luts(self):
        self.path.unlink()
        with gr.Blocks(analytics_enabled=False):
            controls = self.adapter.Script().ui(False)
            self.assertEqual(len(controls), 4)
        tabs = self.adapter.on_ui_tabs()
        self.assertEqual(tabs[0][1], 'LUT Preview')
        self.assertTrue(tabs[0][0].config['dependencies'])

    def test_generation_links_each_original_and_save_callback_retains_link(self):
        p = self.request()
        script = self.adapter.Script()
        def save(image, path, basename, **kw):
            target = self.root / ('original' + kw['suffix'] + '.' + kw['extension'])
            image.save(target)
            self.assertEqual(kw['extension'], 'png')
            return str(target), None
        with patch.object(self.adapter, 'STORE', self.store), patch.object(self.host.images, 'save_image', save):
            script.process(p, True, self.path.name, 1, True)
            pp = SimpleNamespace(image=Image.new('RGB', (2, 2), (10, 80, 200)), index=0)
            script.postprocess_image_after_composite(p, pp)
            token_a = self.store.find(pp.image)
            self.assertEqual(p.extra_generation_params['LUT source'](index=0), token_a)
            qq = SimpleNamespace(image=Image.new('RGB', (2, 2), (20, 90, 210)), index=1)
            script.postprocess_image_after_composite(p, qq)
            token_b = self.store.find(qq.image)
            self.assertNotEqual(token_a, token_b)
            self.assertEqual(p.extra_generation_params['LUT source'](index=0), token_a)
            params = SimpleNamespace(image=pp.image, pnginfo={'parameters': 'prompt\nSeed: 123'})
            pp.image.info = params.pnginfo.copy()  # Forge Extras rewrites metadata.
            self.adapter.before_image_saved(params)
            self.assertEqual(params.pnginfo['Forge LUT source'], token_a)
            self.assertIn('LUT source: ' + token_a, params.pnginfo['parameters'])
            self.assertEqual(pp.image.info['Forge LUT source'], token_a)

    def test_later_iteration_metadata_uses_correct_source_id(self):
        p = self.request()
        p.batch_size, p.iteration = 1, 0
        p.all_prompts = ['first', 'second']
        script = self.adapter.Script()
        script.process(p, True, self.path.name, 1, True)
        p._forge_neo_lut_sources.update({0: 'first-token', 1: 'second-token'})
        p.iteration = 1
        # Forge serializes current-batch metadata with a batch-local index.
        callback = p.extra_generation_params['LUT source']
        self.assertEqual(callback(index=0, all_prompts=p.prompts), 'second-token')
        # Processed.infotext() instead supplies the whole-job array/global index.
        self.assertEqual(callback(index=0, all_prompts=p.all_prompts), 'first-token')

    def test_unsaved_original_still_blocks_metadata_free_graded_upscale(self):
        p = self.request()
        script = self.adapter.Script()
        script.process(p, True, self.path.name, 1, False)
        pp = SimpleNamespace(image=Image.new('RGB', (2, 2), (10, 80, 200)), index=0)
        script.postprocess_image_after_composite(p, pp)
        transferred = pp.image.convert('RGBA')
        transferred.info.clear()
        token = self.store.find(transferred)
        self.assertIsNotNone(token)
        with self.assertRaisesRegex(ValueError, 'original'):
            self.store.resolve(token, self.root)

    def test_extras_script_initializes_after_runner_exists(self):
        host = self.host
        class PostprocessingBase:
            pass
        host.scripts_postprocessing = SimpleNamespace(ScriptPostprocessing=PostprocessingBase)
        host.scripts.scripts_postproc = None  # Forge imports scripts first.
        host.shared.state = SimpleNamespace(skipped=False)
        host.images.read_info_from_image = lambda image: ('', {})
        spec = importlib.util.spec_from_file_location('extras_adapter', Path(__file__).parents[1] / 'scripts' / 'postprocessing_lut.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'modules': host}):
            spec.loader.exec_module(module)
            runner = SimpleNamespace(run=lambda pp, args: None,
                                     scripts_in_preferred_order=lambda: [])
            host.scripts.scripts_postproc = runner
            control = module.LUTExtras()
            self.assertTrue(hasattr(runner, '_forge_lut_original_run'))
            with gr.Blocks(analytics_enabled=False):
                self.assertTrue(control.ui()['enabled'].value)


if __name__ == '__main__':
    unittest.main()
