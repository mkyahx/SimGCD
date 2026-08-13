import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "morph_masks_on_token_grid.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("morph_masks_on_token_grid", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MorphMasksOnTokenGridTests(unittest.TestCase):
    def test_dilation_expands_a_single_foreground_token_by_one_cardinal_ring(self):
        module = load_script_module()
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        dilated = module.apply_morphology(mask, mode="dilate", iterations=1)

        expected = np.zeros((5, 5), dtype=bool)
        expected[2, 2] = True
        expected[1, 2] = True
        expected[3, 2] = True
        expected[2, 1] = True
        expected[2, 3] = True
        np.testing.assert_array_equal(dilated, expected)

    def test_erosion_removes_the_outer_cardinal_ring(self):
        module = load_script_module()
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        eroded = module.apply_morphology(mask, mode="erode", iterations=1)

        expected = np.zeros((5, 5), dtype=bool)
        expected[2, 2] = True
        np.testing.assert_array_equal(eroded, expected)

    def test_process_root_resizes_to_token_grid_and_preserves_relative_paths(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            output_root = temp_root / "output"
            nested = source_root / "class-a"
            nested.mkdir(parents=True)
            source_mask = np.zeros((28, 28), dtype=np.float32)
            source_mask[14:16, 14:16] = 1.0
            source_path = nested / "image.npy"
            np.save(source_path, source_mask)

            count = module.process_mask_root(source_root, output_root, mode="dilate", iterations=1)

            output = np.load(output_root / "class-a" / "image.npy")
            self.assertEqual(count, 1)
            self.assertEqual(output.shape, (14, 14))
            self.assertEqual(output.dtype, np.float32)
            self.assertEqual(int(output.sum()), 5)
            np.testing.assert_array_equal(np.load(source_path), source_mask)


if __name__ == "__main__":
    unittest.main()
