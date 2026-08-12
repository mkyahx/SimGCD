from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class AsymmetricH2ControlTests(unittest.TestCase):
    def test_h2_inverts_the_aligned_tokencut_mask_without_using_c1(self):
        source = (REPO_ROOT / "asymmetric_h2_mask_model.py").read_text(encoding="utf-8")

        self.assertIn("from asymmetric_mask_model import AsymmetricMaskModel", source)
        self.assertIn("class H2BackgroundMaskAsymmetricModel(AsymmetricMaskModel)", source)
        self.assertIn("background_patch_mask = ~foreground_patch_mask", source)
        self.assertNotIn("min_foreground_tokens = 0", source)
        self.assertNotIn("C1FrozenClsAsymmetricMaskModel", source)

    def test_h2_entrypoint_replaces_the_current_asymmetric_model(self):
        source = (REPO_ROOT / "train_asymmetric_h2_repro.py").read_text(encoding="utf-8")

        self.assertIn("from asymmetric_h2_mask_model import H2BackgroundMaskAsymmetricModel", source)
        self.assertIn("asymmetric_mask_model.AsymmetricMaskModel = H2BackgroundMaskAsymmetricModel", source)
        self.assertIn('runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")', source)

    def test_h2_submit_scripts_are_standalone_and_use_k_128(self):
        expected = {"cub": ("cub", 2), "cars": ("scars", 1), "aircraft": ("aircraft", 1)}
        for benchmark, (dataset, memax_weight) in expected.items():
            source = (
                REPO_ROOT / "scripts" / f"submit_asymmetric_h2_{benchmark}_repro.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("python train_asymmetric_h2_repro.py", source)
            self.assertIn(f"--dataset_name '{dataset}'", source)
            self.assertIn("--max_foreground_tokens 128", source)
            self.assertIn("--mask_root /userhome/cs/mkyahx/SimGCD/masks", source)
            self.assertIn("for seed in 0 1 2", source)
            self.assertIn(f"--memax_weight {memax_weight}", source)


if __name__ == "__main__":
    unittest.main()
