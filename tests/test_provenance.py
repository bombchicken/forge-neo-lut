import tempfile
import unittest
from pathlib import Path

from PIL import Image
from test_engine import cube

from forge_neo_lut.engine import grade, load_lut
from forge_neo_lut.provenance import SourceStore


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = SourceStore(self.root / 'data')
        self.lut = load_lut(cube(self.root / 'invert.cube', lambda r, g, b: (1-r, 1-g, 1-b)))
        self.original = Image.new('RGB', (2, 2), (10, 80, 200))
        self.path = self.root / 'original.png'
        self.original.save(self.path)
        self.graded = grade(self.original, self.lut, .5)

    def test_metadata_free_round_trip_and_persistence(self):
        token = self.store.register(self.path, self.lut, .5, self.graded)
        transferred = self.graded.convert('RGBA')
        transferred.info.clear()
        reopened = SourceStore(self.root / 'data')
        self.assertEqual(reopened.find(transferred), token)
        original, lut, strength = reopened.resolve(token, self.root)
        self.assertEqual(original.getpixel((0, 0)), (10, 80, 200))
        self.assertEqual(lut.sha256, self.lut.sha256)
        self.assertEqual(strength, .5)

    def test_missing_or_changed_original_and_lut_rejected(self):
        token = self.store.register(self.path, self.lut, .5, self.graded)
        data = self.path.read_bytes()
        self.path.unlink()
        with self.assertRaises(ValueError):
            self.store.resolve(token, self.root)
        self.path.write_bytes(data + b'changed')
        with self.assertRaises(ValueError):
            self.store.resolve(token, self.root)
        self.path.write_bytes(data)
        cube(self.root / 'invert.cube')
        with self.assertRaisesRegex(ValueError, 'LUT.*changed'):
            self.store.resolve(token, self.root)

    def test_uploaded_paths_are_never_followed(self):
        self.graded.info['Forge LUT source'] = '../../outside.json'
        with self.assertRaises(ValueError):
            self.store.find(self.graded)
        with self.assertRaises(ValueError):
            self.store.resolve('../../outside', self.root)

    def test_legacy_grade_fails_and_unrelated_image_passes(self):
        self.assertIsNone(self.store.find(self.original))
        with self.assertRaisesRegex(ValueError, 'original'):
            self.store.find(self.graded, 'prompt\nSteps: 20, LUT file: invert.cube')

    def test_ambiguous_pixels_require_metadata(self):
        a = self.store.register(self.path, self.lut, .5, self.graded)
        self.store.register(self.path, self.lut, .5, self.graded)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.store.find(self.graded)
        self.graded.info['Forge LUT source'] = a
        self.assertEqual(self.store.find(self.graded), a)

    def test_edited_image_is_not_replaced_using_stale_metadata(self):
        token = self.store.register(self.path, self.lut, .5, self.graded)
        edited = self.graded.copy()
        edited.putpixel((0, 0), (255, 0, 0))
        edited.info['Forge LUT source'] = token
        with self.assertRaisesRegex(ValueError, 'pixels'):
            self.store.find(edited)

    def test_capture_preserves_lossless_upscaled_original(self):
        upscaled = self.original.resize((4, 4))
        upscaled.info['parameters'] = 'prompt\nSeed: 123'
        result = grade(upscaled, self.lut, .5)
        token = self.store.capture(upscaled, self.lut, .5, result)
        source, _, _ = self.store.resolve(token, self.root)
        self.assertEqual(source.size, (4, 4))
        self.assertEqual(source.info['parameters'], 'prompt\nSeed: 123')


if __name__ == '__main__':
    unittest.main()
