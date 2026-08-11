import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class AsymmetricA2ControlTests(unittest.TestCase):
    def test_a2_entrypoint_replaces_only_the_foreground_mask_selector(self):
        entrypoint = REPO_ROOT / "train_asymmetric_a2_repro.py"
        self.assertTrue(entrypoint.exists(), "Missing A2 training entrypoint.")
        source = entrypoint.read_text(encoding="utf-8")

        self.assertIn("A2CardinalityMatchedRandomMaskModel", source)
        self.assertIn("asymmetric_mask_model.AsymmetricMaskModel = A2CardinalityMatchedRandomMaskModel", source)
        self.assertIn('runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")', source)

    def test_a2_model_matches_token_cardinality_without_using_mask_locations(self):
        model_file = REPO_ROOT / "asymmetric_a2_mask_model.py"
        self.assertTrue(model_file.exists(), "Missing A2 cardinality-matched model.")
        source = model_file.read_text(encoding="utf-8")

        self.assertIn("foreground_count = int(aligned_mask[batch_index].sum().item())", source)
        self.assertIn(
            "target_count = min(foreground_count, self.mask_encoder.max_foreground_tokens)",
            source,
        )
        self.assertIn("torch.randperm(token_count", source)
        self.assertIn("random_mask[batch_index, selected_indices] = True", source)
        self.assertNotIn("aligned_mask[batch_index]]", source)

    def test_a2_cars_submit_script_uses_a2_entrypoint_and_token_cut_masks(self):
        script = REPO_ROOT / "scripts" / "submit_asymmetric_a2_cars_repro.sh"
        self.assertTrue(script.exists(), "Missing Cars A2 submit script.")
        source = script.read_text(encoding="utf-8")

        self.assertTrue(source.startswith("#!/bin/bash"))
        self.assertIn("#SBATCH --job-name=asymmetric_a2_cars_repro", source)
        self.assertIn("python train_asymmetric_a2_repro.py", source)
        self.assertIn("--dataset_name 'scars'", source)
        self.assertIn("--mask_root /userhome/cs/mkyahx/SimGCD/masks", source)
        self.assertIn("--max_foreground_tokens 128", source)


if __name__ == "__main__":
    unittest.main()
