import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class AsymmetricMaskSubmitScriptTests(unittest.TestCase):
    def test_submit_scripts_follow_afgcd_convention_and_dataset_presets(self):
        expected = {
            "cub": ("cub", "cub_asymmetric_mask_seed_", "2", "6:00:00"),
            "cars": ("scars", "scars_asymmetric_mask_seed_", "1", "18:00:00"),
            "aircraft": ("aircraft", "aircraft_asymmetric_mask_seed_", "1", "18:00:00"),
        }

        for benchmark, (dataset, exp_prefix, memax, walltime) in expected.items():
            script_path = REPO_ROOT / "scripts" / f"submit_asymmetric_mask_{benchmark}_repro.sh"
            self.assertTrue(script_path.exists(), f"Missing {benchmark} asymmetric-mask submit script.")
            source = script_path.read_text(encoding="utf-8")

            self.assertTrue(source.startswith("#!/bin/bash"))
            self.assertIn(f"#SBATCH --job-name=asymmetric_mask_{benchmark}_repro", source)
            self.assertIn("#SBATCH --partition=batch", source)
            self.assertIn("#SBATCH --gres=gpu:1", source)
            self.assertIn("#SBATCH --mem=24G", source)
            self.assertIn(f"#SBATCH --time={walltime}", source)
            self.assertIn("source /userhome/cs/mkyahx/miniconda3/etc/profile.d/conda.sh", source)
            self.assertIn("conda activate simgcd", source)
            self.assertIn("cd /userhome/cs/mkyahx/SimGCD/", source)
            self.assertIn("export CUBLAS_WORKSPACE_CONFIG=:4096:8", source)
            self.assertIn("for seed in 0 1 2; do", source)
            self.assertIn("PYTHONHASHSEED=$seed CUDA_VISIBLE_DEVICES=0 python train_asymmetric_mask_repro.py", source)
            self.assertIn(f"--dataset_name '{dataset}'", source)
            self.assertIn(f"--memax_weight {memax}", source)
            self.assertIn(f"--exp_name {exp_prefix}${{seed}}", source)
            self.assertIn("--seed $seed", source)
            self.assertIn("--mask_root /userhome/cs/mkyahx/SimGCD/masks", source)
            self.assertIn("--max_foreground_tokens 64", source)
            self.assertIn("conda deactivate", source)


if __name__ == "__main__":
    unittest.main()
