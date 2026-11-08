import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from test_engine import cube

from forge_neo_lut.engine import grade, load_lut
from forge_neo_lut.extras import EXTRAS_NAME, after_saved, install_runner, mark_source
from forge_neo_lut.provenance import SourceStore


class ExtrasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = SourceStore(self.root / 'data')
        self.lut = load_lut(cube(self.root / 'invert.cube', lambda r, g, b: (1-r, 1-g, 1-b)))
        self.original = Image.new('RGB', (2, 2), (10, 80, 200))
        self.path = self.root / 'original.png'
        self.original.save(self.path)
        graded = grade(self.original, self.lut, 1)
        self.token = self.store.register(self.path, self.lut, 1, graded)
        self.pp = SimpleNamespace(image=graded, info={}, extra_images=[])
        self.state = SimpleNamespace(skipped=False)
        self.notices = []
        self.seen = []
        self.control = SimpleNamespace(name=EXTRAS_NAME, args_from=0, args_to=1,
                                       controls={'enabled': None})
        def run(pp, args):
            self.seen.append(pp.image.getpixel((0, 0)))
            pp.image = pp.image.resize((pp.image.width*2, pp.image.height*2))
            self.seen.append('upscaled')
        self.runner = SimpleNamespace(run=run, scripts_in_preferred_order=lambda: [self.control])
        install_runner(self.runner, self.store, self.root, lambda im: im.info.get('parameters', ''),
                       self.state, self.notices.append)

    def test_original_before_pipeline_grade_after_upscale(self):
        self.runner.run(self.pp, [True])
        self.assertEqual(self.seen, [(10, 80, 200), 'upscaled'])
        self.assertEqual(self.pp.image.getpixel((0, 0)), (245, 175, 55))
        self.assertEqual(self.pp.image.size, (4, 4))
        self.assertEqual(self.pp.info['LUT strength'], 1)
        self.assertEqual(self.pp.info['LUT file'], 'invert.cube')

    def test_repeated_extras_uses_upscaled_ungraded_source(self):
        self.runner.run(self.pp, [True])
        token = self.store.find(self.pp.image)
        source, _, _ = self.store.resolve(token, self.root)
        self.assertEqual(source.size, (4, 4))
        self.runner.run(self.pp, [True])
        self.assertEqual(self.pp.image.size, (8, 8))
        self.assertEqual(self.pp.image.getpixel((0, 0)), (245, 175, 55))
        self.assertEqual(self.seen[2], (10, 80, 200))

    def test_missing_original_or_changed_lut_stops_before_upscale(self):
        self.path.unlink()
        self.runner.run(self.pp, [True])
        self.assertTrue(self.state.skipped)
        self.assertFalse(self.seen)
        self.assertTrue(self.notices)

    def test_changed_lut_stops_before_upscale(self):
        cube(self.root / 'invert.cube')
        self.runner.run(self.pp, [True])
        self.assertTrue(self.state.skipped)
        self.assertFalse(self.seen)
        self.assertIn('changed', self.notices[0])

    def test_disabled_and_unrelated_inputs_passthrough(self):
        self.runner.run(self.pp, [False])
        self.assertEqual(self.seen[0], (245, 175, 55))
        pp = SimpleNamespace(image=Image.new('RGB', (2, 2), (0, 0, 0)), info={}, extra_images=[])
        self.runner.run(pp, [True])
        self.assertEqual(pp.image.getpixel((0, 0)), (0, 0, 0))
        self.assertNotIn('LUT file', pp.info)

    def test_wrapper_install_is_idempotent(self):
        install_runner(self.runner, self.store, self.root, lambda im: '', self.state, self.notices.append)
        self.runner.run(self.pp, [True])
        self.assertEqual(self.seen, [(10, 80, 200), 'upscaled'])

    def test_secondary_outputs_are_graded_and_linked_individually(self):
        base = self.runner._forge_lut_original_run
        def extra(pp, args):
            base(pp, args)
            pp.extra_images.append(SimpleNamespace(image=Image.new('RGB', (3, 3), (20, 90, 210)), info={}))
        self.runner._forge_lut_original_run = extra
        self.runner.run(self.pp, [True])
        secondary = self.pp.extra_images[0]
        self.assertEqual(secondary.image.getpixel((0, 0)), (235, 165, 45))
        token = self.store.find(secondary.image)
        source, _, _ = self.store.resolve(token, self.root)
        self.assertEqual(source.getpixel((0, 0)), (20, 90, 210))

    def test_encoded_jpeg_pixels_can_be_matched_without_metadata(self):
        # A saved JPEG differs from the pre-encoding pixels. The callback must
        # index the actual encoded pixels used by Forge's gallery.
        path = self.root / 'graded.jpg'
        image = mark_source(self.pp.image, self.token)
        image.save(path, quality=80)
        after_saved(SimpleNamespace(image=image, filename=str(path)), self.store)
        with Image.open(path) as uploaded:
            uploaded.info.clear()
            self.assertEqual(self.store.find(uploaded), self.token)

    def test_outer_extras_metadata_reset_without_saving_uses_current_pixels(self):
        self.runner.run(self.pp, [True])
        current_token = self.store.find(self.pp.image)
        transferred = self.pp.image.copy()  # PIL copies do not retain private attributes.
        # Forge does this after the runner, even with save_output=False.
        transferred.info = {'Forge LUT source': self.token}
        self.assertEqual(self.store.find(transferred), current_token)


if __name__ == '__main__':
    unittest.main()
