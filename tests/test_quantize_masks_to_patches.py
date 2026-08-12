import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "quantize_masks_to_patches.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("quantize_masks_to_patches", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantizeMasksToPatchesTests(unittest.TestCase):
    def test_quantizes_each_16_by_16_block_using_half_coverage_threshold(self):
        module = load_script_module()
        mask = np.zeros((17, 32), dtype=np.float32)
        mask[:8, :16] = 1.0  # Exactly half: foreground.
        mask[:7, 16:] = 1.0  # Below half: background.
        mask[16:, :16] = 1.0  # Partial edge block: all foreground.

        quantized = module.quantize_mask_to_patches(mask, patch_size=16)

        self.assertEqual(quantized.shape, mask.shape)
        self.assertTrue(np.all(quantized[:16, :16] == 1.0))
        self.assertTrue(np.all(quantized[:16, 16:] == 0.0))
        self.assertTrue(np.all(quantized[16:, :16] == 1.0))
        self.assertTrue(np.all(quantized[16:, 16:] == 0.0))

    def test_process_root_preserves_relative_paths_and_leaves_source_unchanged(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            output_root = temp_root / "output"
            nested = source_root / "class-a"
            nested.mkdir(parents=True)
            source_mask = np.ones((16, 16), dtype=np.float32)
            source_path = nested / "image.npy"
            np.save(source_path, source_mask)

            count = module.process_mask_root(source_root, output_root, patch_size=16)

            self.assertEqual(count, 1)
            self.assertTrue(np.array_equal(np.load(source_path), source_mask))
            self.assertTrue(np.array_equal(np.load(output_root / "class-a" / "image.npy"), source_mask))


if __name__ == "__main__":
    unittest.main()
