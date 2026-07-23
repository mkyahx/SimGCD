import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class GLSimMaskLauncherTests(unittest.TestCase):
    def test_cars_and_aircraft_launchers_preserve_their_original_baseline_presets(self):
        expected = {
            "cars": ("scars", "scars_glsim_mask_repro"),
            "aircraft": ("aircraft", "aircraft_glsim_mask_repro"),
        }

        for benchmark, (dataset_name, exp_name) in expected.items():
            launcher_path = REPO_ROOT / "scripts" / f"run_glsim_mask_{benchmark}_repro.sh"
            self.assertTrue(launcher_path.exists(), f"Missing {benchmark} GLSim-mask launcher.")
            source = launcher_path.read_text(encoding="utf-8")

            self.assertIn("SEED=1", source)
            self.assertIn('MASK_ROOT="/path/to/tokencut/masks"', source)
            self.assertIn("MAX_FOREGROUND_TOKENS=64", source)
            self.assertIn("export PYTHONHASHSEED=\"$SEED\"", source)
            self.assertIn("export CUBLAS_WORKSPACE_CONFIG=:4096:8", source)
            self.assertIn("python train_glsim_mask_repro.py", source)
            self.assertIn(f"--dataset_name '{dataset_name}'", source)
            self.assertIn("--batch_size 128", source)
            self.assertIn("--grad_from_block 11", source)
            self.assertIn("--epochs 200", source)
            self.assertIn("--num_workers 8", source)
            self.assertIn("--use_ssb_splits", source)
            self.assertIn("--sup_weight 0.35", source)
            self.assertIn("--weight_decay 5e-5", source)
            self.assertIn("--transform 'imagenet'", source)
            self.assertIn("--lr 0.1", source)
            self.assertIn("--eval_funcs 'v2'", source)
            self.assertIn("--warmup_teacher_temp 0.07", source)
            self.assertIn("--teacher_temp 0.04", source)
            self.assertIn("--warmup_teacher_temp_epochs 30", source)
            self.assertIn("--memax_weight 1", source)
            self.assertIn(f"--exp_name {exp_name}", source)
            self.assertIn('--seed "$SEED"', source)
            self.assertIn('--mask_root "$MASK_ROOT"', source)
            self.assertIn('"${K_ARGS[@]}"', source)
            self.assertNotIn('"$@"', source)


if __name__ == "__main__":
    unittest.main()
