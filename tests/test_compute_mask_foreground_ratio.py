import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compute_mask_foreground_ratio.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("compute_mask_foreground_ratio", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ComputeMaskForegroundRatioTests(unittest.TestCase):
    def test_compute_statistics_uses_training_threshold_and_recurses(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "first.npy", np.array([[0.0, 0.5], [0.51, 1.0]], dtype=np.float32))
            nested = root / "nested"
            nested.mkdir()
            np.save(nested / "second.npy", np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32))

            rows, summary = module.compute_foreground_ratios(root)

        self.assertEqual([row["path"] for row in rows], ["first.npy", "nested/second.npy"])
        self.assertAlmostEqual(rows[0]["foreground_ratio"], 0.5)
        self.assertAlmostEqual(rows[1]["foreground_ratio"], 0.5)
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["mean"], 0.5)
        self.assertAlmostEqual(summary["std"], 0.0)

    def test_empty_root_fails_with_clear_error(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "No .npy mask files"):
                module.compute_foreground_ratios(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
