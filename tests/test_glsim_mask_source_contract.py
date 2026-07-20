import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class GLSimMaskSourceContractTests(unittest.TestCase):
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
