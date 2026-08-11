from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class AsymmetricB2C1ControlTests(unittest.TestCase):
    def test_b2_model_routes_both_training_views_through_foreground_encoding(self):
        source = (REPO_ROOT / "asymmetric_b2_mask_model.py").read_text(encoding="utf-8")
        self.assertIn("foreground_one = self._foreground_features(view_one, mask_one)", source)
        self.assertIn("foreground_two = self._foreground_features(view_two, mask_two)", source)
        self.assertIn("torch.cat([foreground_one, foreground_two], dim=0)", source)

    def test_b2_trainer_imports_the_b2_model(self):
        source = (REPO_ROOT / "train_asymmetric_b2_repro.py").read_text(encoding="utf-8")
        self.assertIn("from asymmetric_b2_mask_model import B2ForegroundPairMaskModel", source)
        self.assertIn("model = B2ForegroundPairMaskModel(", source)

    def test_b2_entrypoint_forwards_both_patch_masks(self):
        source = (REPO_ROOT / "train_asymmetric_b2_repro.py").read_text(encoding="utf-8")
        self.assertIn(
            "student((images[0], images[1], patch_mask[0], patch_mask[1]))",
            source,
        )

    def test_b2_submit_scripts_use_the_benchmark_presets_and_k_128(self):
        expected = {"cub": "cub", "cars": "scars", "aircraft": "aircraft"}
        for benchmark, dataset in expected.items():
            source = (
                REPO_ROOT / "scripts" / f"submit_asymmetric_b2_{benchmark}_repro.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("python train_asymmetric_b2_repro.py", source)
            self.assertIn(f"--dataset_name '{dataset}'", source)
            self.assertIn("--max_foreground_tokens 128", source)

    def test_c1_uses_a_frozen_dino_cls_buffer_not_a_parameter(self):
        source = (REPO_ROOT / "asymmetric_c1_mask_model.py").read_text(encoding="utf-8")
        self.assertIn("self.register_buffer(\"foreground_cls_token\"", source)
        self.assertIn("backbone.cls_token.detach().clone()", source)
        self.assertNotIn("nn.Parameter", source)
        self.assertIn("foreground_cls = self.foreground_cls_token", source)

    def test_c1_entrypoint_patches_only_the_model_class(self):
        source = (REPO_ROOT / "train_asymmetric_c1_repro.py").read_text(encoding="utf-8")
        self.assertIn(
            "asymmetric_mask_model.AsymmetricMaskModel = C1FrozenClsAsymmetricMaskModel",
            source,
        )
        self.assertIn(
            'runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")',
            source,
        )
