import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from forge_neo_lut.engine import annotate, grade, load_lut, save_preview


def cube(path, transform=lambda r, g, b: (r, g, b), header=''):
    rows = ['TITLE "Test"', 'LUT_3D_SIZE 2', header]
    for b in (0, 1):
        for g in (0, 1):
            for r in (0, 1):
                rows.append(' '.join(map(str, transform(r, g, b))))
    path.write_text('\n'.join(rows), encoding='utf-8')
    return path


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'test.cube'

    def test_identity_preserves_all_channels_and_source(self):
        lut = load_lut(cube(self.path))
        pixels = np.random.default_rng(3).integers(0, 256, (513, 9, 3), dtype=np.uint8)
        source = Image.fromarray(pixels)
        result = grade(source, lut, 1)
        np.testing.assert_array_equal(np.array(result), pixels)
        np.testing.assert_array_equal(np.array(source), pixels)

    def test_red_fastest_order_and_fractional_interpolation(self):
        lut = load_lut(cube(self.path, lambda r, g, b: (b, r, g)))
        source = Image.new('RGB', (1, 1), (51, 102, 204))
        self.assertEqual(grade(source, lut, 1).getpixel((0, 0)), (204, 51, 102))

    def test_nonlinear_corner_interpolation(self):
        lut = load_lut(cube(self.path, lambda r, g, b: (r*g*b, g, b)))
        source = Image.new('RGB', (1, 1), (128, 128, 128))
        self.assertEqual(grade(source, lut, 1).getpixel((0, 0)), (32, 128, 128))

    def test_strength_and_alpha(self):
        lut = load_lut(cube(self.path, lambda r, g, b: (1-r, 1-g, 1-b)))
        source = Image.new('RGBA', (1, 1), (0, 100, 255, 42))
        self.assertEqual(grade(source, lut, 0).getpixel((0, 0)), (0, 100, 255, 42))
        self.assertEqual(grade(source, lut, .5).getpixel((0, 0)), (128, 128, 128, 42))
        self.assertEqual(grade(source, lut, 1).getpixel((0, 0)), (255, 155, 0, 42))
        for strength in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                grade(source, lut, strength)

    def test_domains_and_clipping(self):
        lut = load_lut(cube(self.path, header='DOMAIN_MIN 0.2 0.2 0.2\nDOMAIN_MAX 0.8 0.8 0.8'))
        self.assertEqual(grade(Image.new('RGB', (1, 1), (0, 102, 255)), lut, 1).getpixel((0, 0)), (0, 85, 255))

    def test_content_hash_changes_even_same_size(self):
        lut = load_lut(cube(self.path))
        self.assertEqual(lut.sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        other = load_lut(cube(self.path, lambda r, g, b: (b, g, r)))
        self.assertNotEqual(other.sha256, lut.sha256)

    def test_invalid_and_unsupported_luts(self):
        valid = cube(self.path).read_text()
        cases = [valid.replace('LUT_3D_SIZE 2', 'LUT_1D_SIZE 2'),
                 valid + '\nLUT_1D_SIZE 2', valid + '\n0 0 0',
                 valid.replace('0 0 0', 'nan 0 0', 1),
                 valid.replace('0 0 0', 'inf 0 0', 1),
                 valid.replace('LUT_3D_SIZE 2', 'LUT_3D_SIZE 1'),
                 valid + '\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 0 0 0',
                 valid + '\nUNKNOWN 1', 'LUT_3D_SIZE 2\n0 0 0',
                 valid + '\nLUT_3D_SIZE 2']
        for value in cases:
            with self.subTest(value=value):
                self.path.write_text(value)
                with self.assertRaises(ValueError):
                    load_lut(self.path)

    def test_metadata_survives_png_round_trip(self):
        lut = load_lut(cube(self.path))
        source = Image.new('RGB', (2, 2))
        source.info = {'parameters': 'a cat\nSteps: 20, Seed: 123', 'custom': 'hello',
                       'icc_profile': b'profile', 'dpi': (72, 72)}
        result = annotate(grade(source, lut, .75), lut, .75)
        output = save_preview(result)
        self.addCleanup(lambda: Path(output).unlink(missing_ok=True))
        with Image.open(output) as reopened:
            self.assertEqual(reopened.info['parameters'], source.info['parameters'])
            self.assertEqual(reopened.info['custom'], 'hello')
            self.assertEqual(reopened.info['icc_profile'], b'profile')
            self.assertEqual(json.loads(reopened.info['Forge LUT'])['sha256'], lut.sha256)
            self.assertEqual(json.loads(reopened.info['Forge LUT'])['strength'], .75)
        self.assertNotIn('Forge LUT', source.info)


if __name__ == '__main__':
    unittest.main()
