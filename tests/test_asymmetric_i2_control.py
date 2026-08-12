from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class AsymmetricI2ControlTests(unittest.TestCase):
    def test_i2_forces_all_ones_masks_and_reuses_current_asymmetric_trainer(self):
        source = (REPO_ROOT / "train_asymmetric_i2_repro.py").read_text(encoding="utf-8")

        self.assertIn("from data.mask_dataset import DatasetWithPatchMask", source)
        self.assertIn("return np.ones((14, 14), dtype=np.float32)", source)
        self.assertIn("DatasetWithPatchMask._get_mask = _all_ones_mask", source)
        self.assertIn('runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")', source)
        self.assertNotIn("C1FrozenClsAsymmetricMaskModel", source)

    def test_i2_submit_scripts_are_standalone_asym_controls_with_k_128(self):
        expected = {"cub": ("cub", 2), "cars": ("scars", 1), "aircraft": ("aircraft", 1)}
        for benchmark, (dataset, memax_weight) in expected.items():
            source = (
                REPO_ROOT / "scripts" / f"submit_asymmetric_i2_{benchmark}_repro.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("python train_asymmetric_i2_repro.py", source)
            self.assertIn(f"--dataset_name '{dataset}'", source)
            self.assertIn("--max_foreground_tokens 128", source)
            self.assertIn("for seed in 0 1 2", source)
            self.assertIn(f"--memax_weight {memax_weight}", source)
            self.assertNotIn("--mask_root /userhome/cs/mkyahx/SimGCD/masks", source)


if __name__ == "__main__":
    unittest.main()
