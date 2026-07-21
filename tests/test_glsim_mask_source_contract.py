import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class GLSimMaskSourceContractTests(unittest.TestCase):
    def test_repro_entrypoint_enforces_strict_gpu_and_dataloader_determinism(self):
        repro_path = REPO_ROOT / "train_glsim_mask_repro.py"
        self.assertTrue(repro_path.exists(), "The strict reproducibility entrypoint is missing.")
        source = repro_path.read_text(encoding="utf-8")

        self.assertLess(
            source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]'),
            source.index("import torch"),
        )
        self.assertIn('parser.add_argument("--seed", required=True, type=int)', source)
        self.assertIn("def seed_worker(worker_id):", source)
        self.assertIn("torch.use_deterministic_algorithms(True)", source)
        self.assertNotIn("torch.use_deterministic_algorithms(True, warn_only=True)", source)
        self.assertIn("torch.backends.cudnn.benchmark = False", source)
        self.assertIn("torch.backends.cuda.matmul.allow_tf32 = False", source)
        self.assertIn("torch.backends.cudnn.allow_tf32 = False", source)
        self.assertIn("generator=train_generator", source)
        self.assertIn("worker_init_fn=seed_worker", source)
        self.assertIn('parser.add_argument("--max_foreground_tokens", default=None, type=int)', source)
        self.assertIn("max_foreground_tokens=args.max_foreground_tokens", source)

    def test_repro_launcher_exports_process_level_seed_configuration(self):
        launcher_path = REPO_ROOT / "scripts" / "run_glsim_mask_repro.sh"
        self.assertTrue(launcher_path.exists(), "The reproducible launcher is missing.")
        source = launcher_path.read_text(encoding="utf-8")

        self.assertIn("SEED=1", source)
        self.assertIn('MASK_ROOT="/path/to/tokencut/masks"', source)
        self.assertIn("MAX_FOREGROUND_TOKENS=64", source)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", source)
        self.assertIn('export PYTHONHASHSEED="$SEED"', source)
        self.assertIn("export CUBLAS_WORKSPACE_CONFIG=:4096:8", source)
        self.assertIn("python train_glsim_mask_repro.py \\", source)
        self.assertIn('--seed "$SEED"', source)
        self.assertIn('--mask_root "$MASK_ROOT"', source)
        self.assertIn('--max_foreground_tokens "$MAX_FOREGROUND_TOKENS"', source)
        self.assertNotIn('"$@"', source)

    def test_model_uses_batched_padding_and_masked_attention(self):
        source = (REPO_ROOT / "glsim_mask_model.py").read_text(encoding="utf-8")

        self.assertIn("def _pack_foreground_tokens", source)
        self.assertIn("def _masked_dino_block", source)
        self.assertIn("attn.masked_fill(~key_mask", source)
        self.assertIn("x = x * valid_mask.unsqueeze(-1).to(x.dtype)", source)

    def test_model_defines_global_foreground_fusion_path(self):
        source = (REPO_ROOT / "glsim_mask_model.py").read_text(encoding="utf-8")

        self.assertIn("class MaskForegroundGLSimModel", source)
        self.assertIn("global_cls = self.backbone(images)", source)
        self.assertIn("self.foreground_cls_token", source)
        self.assertIn("self.backbone.pos_embed[:, :1]", source)
        self.assertIn("foreground_cls = self._foreground_cls_batched(patch_tokens, patch_mask)", source)
        self.assertIn("two_cls_tokens = torch.stack([global_cls, foreground_cls], dim=1)", source)
        self.assertIn("fused_cls = self.fusion_norm(fused_tokens[:, 0])", source)
        self.assertIn("return self.head(fused_cls)", source)

    def test_training_entrypoint_is_additive_and_uses_mask_root(self):
        source = (REPO_ROOT / "train_glsim_mask.py").read_text(encoding="utf-8")

        self.assertIn("from glsim_mask_model import MaskForegroundGLSimModel", source)
        self.assertIn("parser.add_argument('--mask_root'", source)
        self.assertIn("DatasetWithPatchMask", source)
        self.assertIn("PairedMaskViewGenerator", source)
        self.assertIn("student_proj, student_out = student((images, patch_mask))", source)


if __name__ == "__main__":
    unittest.main()
