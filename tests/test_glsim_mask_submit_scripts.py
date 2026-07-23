import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class GLSimMaskSubmitScriptTests(unittest.TestCase):
    def test_submit_scripts_follow_afgcd_slurm_convention_and_glsim_presets(self):
        expected = {
            "cub": {
                "dataset": "cub",
                "exp_prefix": "cub_glsim_mask_seed_",
                "memax_weight": "2",
                "time": "6:00:00",
            },
            "cars": {
                "dataset": "scars",
                "exp_prefix": "scars_glsim_mask_seed_",
                "memax_weight": "1",
                "time": "18:00:00",
            },
            "aircraft": {
                "dataset": "aircraft",
                "exp_prefix": "aircraft_glsim_mask_seed_",
                "memax_weight": "1",
                "time": "18:00:00",
            },
        }

        for benchmark, config in expected.items():
            script_path = REPO_ROOT / "scripts" / f"submit_glsim_mask_{benchmark}_repro.sh"
            self.assertTrue(script_path.exists(), f"Missing {benchmark} GLSim-mask submit script.")
            source = script_path.read_text(encoding="utf-8")

            self.assertTrue(source.startswith("#!/bin/bash"))
            self.assertIn(f"#SBATCH --job-name=glsim_mask_{benchmark}_repro", source)
            self.assertIn("#SBATCH --partition=batch", source)
            self.assertIn("#SBATCH --nodes=1", source)
            self.assertIn("#SBATCH --ntasks-per-node=1", source)
            self.assertIn("#SBATCH --cpus-per-task=4", source)
            self.assertIn("#SBATCH --gres=gpu:1", source)
            self.assertIn("#SBATCH --mem=24G", source)
            self.assertIn(f"#SBATCH --time={config['time']}", source)
            self.assertIn("#SBATCH --mail-type=BEGIN,END,FAIL", source)
            self.assertIn("#SBATCH --mail-user=mkyahx@connect.hku.hk", source)
            self.assertIn("source /userhome/cs/mkyahx/miniconda3/etc/profile.d/conda.sh", source)
            self.assertIn("conda activate simgcd", source)
            self.assertIn("cd /userhome/cs/mkyahx/SimGCD/", source)
            self.assertIn("for seed in 0 1 2; do", source)
            self.assertIn("python train_glsim_mask_repro.py", source)
            self.assertIn(f"--dataset_name '{config['dataset']}'", source)
            self.assertIn(f"--memax_weight {config['memax_weight']}", source)
            self.assertIn(f"--exp_name {config['exp_prefix']}${{seed}}", source)
            self.assertIn("--seed $seed", source)
            self.assertIn("--mask_root /userhome/cs/mkyahx/SimGCD/masks", source)
            self.assertIn("--max_foreground_tokens 64", source)
            self.assertIn("conda deactivate", source)
            self.assertIn('echo "Finish"', source)


if __name__ == "__main__":
    unittest.main()
