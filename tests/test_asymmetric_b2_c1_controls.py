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
