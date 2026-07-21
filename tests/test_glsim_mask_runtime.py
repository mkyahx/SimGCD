import unittest

import torch
import torch.nn as nn

from glsim_mask_model import MaskForegroundGLSimModel


class _IdentityBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.drop_path = nn.Identity()
        self.mlp = nn.Identity()
        self.attn = nn.Module()
        self.attn.num_heads = 1
        self.attn.scale = 1.0
        self.attn.qkv = nn.Linear(2, 6, bias=False)
        self.attn.attn_drop = nn.Identity()
        self.attn.proj = nn.Identity()
        self.attn.proj_drop = nn.Identity()
        with torch.no_grad():
            self.attn.qkv.weight.copy_(torch.cat([torch.eye(2), torch.eye(2), torch.eye(2)], dim=0))


class BatchedForegroundRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.model = MaskForegroundGLSimModel(backbone=None, head=nn.Identity(), feat_dim=2, fusion_heads=1)

    def test_packing_pads_each_sample_to_the_batch_foreground_length(self):
        patch_tokens = torch.tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            ]
        )
        patch_mask = torch.tensor([[True, False, True], [False, True, False]])

        packed, valid_mask = self.model._pack_foreground_tokens(patch_tokens, patch_mask)

        self.assertEqual(tuple(packed.shape), (2, 2, 2))
        self.assertTrue(torch.equal(valid_mask, torch.tensor([[True, True], [True, False]])))
        self.assertTrue(torch.equal(packed[0], torch.tensor([[1.0, 1.0], [3.0, 3.0]])))
        self.assertTrue(torch.equal(packed[1, 0], torch.tensor([5.0, 5.0])))
        self.assertTrue(torch.equal(packed[1, 1], torch.zeros(2)))

    def test_packing_caps_foreground_tokens_with_deterministic_uniform_selection(self):
        capped_model = MaskForegroundGLSimModel(
            backbone=None,
            head=nn.Identity(),
            feat_dim=2,
            fusion_heads=1,
            max_foreground_tokens=2,
        )
        patch_tokens = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0]],
            ]
        )
        patch_mask = torch.tensor([[True, True, True, True, True], [True, False, True, False, True]])

        packed, valid_mask = capped_model._pack_foreground_tokens(patch_tokens, patch_mask)

        self.assertEqual(tuple(packed.shape), (2, 2, 2))
        self.assertTrue(torch.equal(valid_mask, torch.ones(2, 2, dtype=torch.bool)))
        self.assertTrue(torch.equal(packed[0], torch.tensor([[0.0, 0.0], [4.0, 4.0]])))
        self.assertTrue(torch.equal(packed[1], torch.tensor([[10.0, 10.0], [14.0, 14.0]])))

    def test_masked_block_ignores_values_in_padded_positions(self):
        block = _IdentityBlock()
        valid_mask = torch.tensor([[True, True, False]])
        tokens_a = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [7.0, -4.0]]])
        tokens_b = tokens_a.clone()
        tokens_b[:, 2] = torch.tensor([10000.0, -10000.0])

        output_a = self.model._masked_dino_block(block, tokens_a, valid_mask)
        output_b = self.model._masked_dino_block(block, tokens_b, valid_mask)

        self.assertTrue(torch.allclose(output_a[:, :2], output_b[:, :2], atol=1e-6, rtol=0.0))
        self.assertTrue(torch.equal(output_a[:, 2], torch.zeros_like(output_a[:, 2])))
        self.assertTrue(torch.equal(output_b[:, 2], torch.zeros_like(output_b[:, 2])))


if __name__ == "__main__":
    unittest.main()
