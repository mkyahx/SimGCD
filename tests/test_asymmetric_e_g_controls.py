from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class AsymmetricDirectionalAndInferenceControlTests(unittest.TestCase):
    def test_e2_and_e3_directional_losses_use_opposite_teacher_student_pairs(self):
        source = (REPO_ROOT / "directional_distill_loss.py").read_text(encoding="utf-8")

        self.assertIn("class GlobalToForegroundDistillLoss", source)
        self.assertIn("teacher_out[0]", source)
        self.assertIn("student_out[1]", source)
        self.assertIn("class ForegroundToGlobalDistillLoss", source)
        self.assertIn("teacher_out[1]", source)
        self.assertIn("student_out[0]", source)

    def test_e2_and_e3_entrypoints_replace_only_distill_loss(self):
        expected = {
            "e2": "GlobalToForegroundDistillLoss",
            "e3": "ForegroundToGlobalDistillLoss",
        }
        for variant, loss_class in expected.items():
            source = (REPO_ROOT / f"train_asymmetric_{variant}_repro.py").read_text(encoding="utf-8")
            self.assertIn(f"model.DistillLoss = {loss_class}", source)
            self.assertIn("from asymmetric_c1_mask_model import C1FrozenClsAsymmetricMaskModel", source)
            self.assertIn(
                "asymmetric_mask_model.AsymmetricMaskModel = C1FrozenClsAsymmetricMaskModel",
                source,
            )
            self.assertIn('runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")', source)

    def test_g2_and_g3_preserve_training_and_change_only_eval_features(self):
        g2_source = (REPO_ROOT / "asymmetric_g2_mask_model.py").read_text(encoding="utf-8")
        self.assertIn("from asymmetric_c1_mask_model import C1FrozenClsAsymmetricMaskModel", g2_source)
        self.assertIn("class GlobalOnlyInferenceAsymmetricMaskModel(C1FrozenClsAsymmetricMaskModel)", g2_source)
        self.assertIn("if self.training:", g2_source)
        self.assertIn("return super().forward(inputs)", g2_source)
        self.assertIn("global_cls = self.mask_encoder.backbone(images)", g2_source)
        self.assertIn("return self.head(global_cls)", g2_source)

        g3_source = (REPO_ROOT / "asymmetric_g3_mask_model.py").read_text(encoding="utf-8")
        self.assertIn("from asymmetric_c1_mask_model import C1FrozenClsAsymmetricMaskModel", g3_source)
        self.assertIn("class MeanFeatureInferenceAsymmetricMaskModel(C1FrozenClsAsymmetricMaskModel)", g3_source)
        self.assertIn("if self.training:", g3_source)
        self.assertIn("return super().forward(inputs)", g3_source)
        self.assertIn("foreground_cls = self._foreground_features(images, patch_mask)", g3_source)
        self.assertIn("fused_cls = 0.5 * (global_cls + foreground_cls)", g3_source)

    def test_g2_and_g3_entrypoints_replace_only_the_model_class(self):
        expected = {
            "g2": "GlobalOnlyInferenceAsymmetricMaskModel",
            "g3": "MeanFeatureInferenceAsymmetricMaskModel",
        }
        for variant, model_class in expected.items():
            source = (REPO_ROOT / f"train_asymmetric_{variant}_repro.py").read_text(encoding="utf-8")
            self.assertIn(f"asymmetric_mask_model.AsymmetricMaskModel = {model_class}", source)
            self.assertIn('runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")', source)

    def test_all_e_and_g_scripts_keep_benchmark_presets_and_k_128(self):
        datasets = {"cub": "cub", "cars": "scars", "aircraft": "aircraft"}
        memax = {"cub": 2, "cars": 1, "aircraft": 1}
        for variant in ("e2", "e3", "g2", "g3"):
            for benchmark, dataset in datasets.items():
                script = REPO_ROOT / "scripts" / f"submit_asymmetric_{variant}_{benchmark}_repro.sh"
                source = script.read_text(encoding="utf-8")
                self.assertIn(f"python train_asymmetric_{variant}_repro.py", source)
                self.assertIn(f"--dataset_name '{dataset}'", source)
                self.assertIn("--max_foreground_tokens 128", source)
                self.assertIn("--mask_root /userhome/cs/mkyahx/SimGCD/masks", source)
                self.assertIn("for seed in 0 1 2", source)
                self.assertIn(f"--memax_weight {memax[benchmark]}", source)
                self.assertNotIn("run_asymmetric_variant_repro.sh", source)


if __name__ == "__main__":
    unittest.main()
