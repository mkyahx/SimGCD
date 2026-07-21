import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoCLSFusion(nn.Module):
    """GLSim-style one-block Transformer Aggregator over global and foreground CLS tokens."""

    def __init__(self, dim, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, global_cls, foreground_cls):
        two_cls_tokens = torch.stack([global_cls, foreground_cls], dim=1)
        fused_tokens = self.block(two_cls_tokens)
        fused_cls = self.fusion_norm(fused_tokens[:, 0])
        return fused_cls


class MaskForegroundGLSimModel(nn.Module):
    """
    SimGCD-compatible wrapper with:
      1. original DINO global CLS path;
      2. mask-filtered foreground-token path with a new CLS token;
      3. GLSim-style two-CLS fusion before the original SimGCD head.
    """

    def __init__(
        self,
        backbone,
        head,
        feat_dim=768,
        fusion_heads=12,
        min_foreground_tokens=1,
        max_foreground_tokens=None,
    ):
        super().__init__()
        if max_foreground_tokens is not None and max_foreground_tokens < 1:
            raise ValueError("max_foreground_tokens must be positive when provided.")
        self.backbone = backbone
        self.head = head
        self.feat_dim = feat_dim
        self.min_foreground_tokens = min_foreground_tokens
        self.max_foreground_tokens = max_foreground_tokens
        self.foreground_cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.fusion = TwoCLSFusion(dim=feat_dim, num_heads=fusion_heads)
        self._init_foreground_cls()

    def _init_foreground_cls(self):
        with torch.no_grad():
            if hasattr(self.backbone, "cls_token") and self.backbone.cls_token.shape[-1] == self.feat_dim:
                self.foreground_cls_token.copy_(self.backbone.cls_token.detach().clone())
            else:
                nn.init.trunc_normal_(self.foreground_cls_token, std=0.02)

    def _align_patch_mask(self, patch_mask, batch_size, token_count, device):
        if patch_mask is None:
            return torch.ones(batch_size, token_count, device=device, dtype=torch.bool)

        patch_mask = patch_mask.to(device=device, dtype=torch.float32)
        if patch_mask.dim() == 4 and patch_mask.shape[1] == 1:
            patch_mask = patch_mask.squeeze(1)
        if patch_mask.dim() == 3:
            patch_mask = patch_mask.unsqueeze(1)
        elif patch_mask.dim() == 2:
            if patch_mask.shape[-1] == token_count:
                return patch_mask > 0.5
            side = int(patch_mask.shape[-1] ** 0.5)
            if side * side != patch_mask.shape[-1]:
                raise ValueError(f"Cannot reshape patch mask with shape {tuple(patch_mask.shape)}.")
            patch_mask = patch_mask.view(batch_size, 1, side, side)
        else:
            raise ValueError(f"Unsupported patch mask shape: {tuple(patch_mask.shape)}")

        target_side = int(token_count ** 0.5)
        if target_side * target_side != token_count:
            raise ValueError(f"Token count {token_count} is not a square patch grid.")

        if patch_mask.shape[-2:] != (target_side, target_side):
            patch_mask = F.interpolate(patch_mask, size=(target_side, target_side), mode="nearest")

        return patch_mask.squeeze(1).reshape(batch_size, token_count) > 0.5

    def _prepare_patch_tokens(self, images):
        prepared_tokens = self.backbone.prepare_tokens(images)
        return prepared_tokens[:, 1:]

    def _foreground_cls_with_position(self):
        foreground_cls = self.foreground_cls_token
        if hasattr(self.backbone, "pos_embed") and self.backbone.pos_embed.shape[-1] == self.feat_dim:
            foreground_cls = foreground_cls + self.backbone.pos_embed[:, :1]
        return foreground_cls

    def _pack_foreground_tokens(self, patch_tokens, patch_mask):
        """Pack variable-length foreground token sets into one padded batch.

        Samples with too few selected tokens retain their complete patch grid.  This
        preserves a valid foreground stream for empty or degenerate masks while
        keeping the normal path strictly limited to foreground tokens.
        """
        effective_mask = patch_mask.clone()
        too_small = effective_mask.sum(dim=1) < self.min_foreground_tokens
        if too_small.any():
            effective_mask[too_small] = True

        foreground_lengths = effective_mask.sum(dim=1)
        if self.max_foreground_tokens is not None:
            foreground_lengths = foreground_lengths.clamp(max=self.max_foreground_tokens)
        max_foreground_length = int(foreground_lengths.max().item())
        batch_size, _, feature_dim = patch_tokens.shape
        packed_tokens = patch_tokens.new_zeros(batch_size, max_foreground_length, feature_dim)
        valid_mask = torch.zeros(
            batch_size,
            max_foreground_length,
            device=patch_tokens.device,
            dtype=torch.bool,
        )

        for batch_index in range(batch_size):
            selected_tokens = patch_tokens[batch_index][effective_mask[batch_index]]
            if self.max_foreground_tokens is not None and selected_tokens.shape[0] > self.max_foreground_tokens:
                uniform_positions = torch.linspace(
                    0,
                    selected_tokens.shape[0] - 1,
                    steps=self.max_foreground_tokens,
                    device=patch_tokens.device,
                ).round().to(dtype=torch.long)
                selected_tokens = selected_tokens.index_select(0, uniform_positions)
            selected_length = selected_tokens.shape[0]
            packed_tokens[batch_index, :selected_length] = selected_tokens
            valid_mask[batch_index, :selected_length] = True

        return packed_tokens, valid_mask

    @staticmethod
    def _zero_invalid_tokens(x, valid_mask):
        return x * valid_mask.unsqueeze(-1).to(x.dtype)

    def _masked_dino_attention(self, attn_module, x, valid_mask):
        """DINO-v1 attention that cannot read padding keys or retain padded queries."""
        batch_size, token_count, feature_dim = x.shape
        head_dim = feature_dim // attn_module.num_heads
        qkv = attn_module.qkv(x).reshape(
            batch_size,
            token_count,
            3,
            attn_module.num_heads,
            head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn = (query @ key.transpose(-2, -1)) * attn_module.scale
        key_mask = valid_mask[:, None, None, :]
        attn = attn.masked_fill(~key_mask, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        attn = attn.masked_fill(~key_mask, 0.0)
        attn = attn * valid_mask[:, None, :, None].to(attn.dtype)
        attn = attn_module.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(batch_size, token_count, feature_dim)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return self._zero_invalid_tokens(x, valid_mask)

    def _masked_dino_block(self, block, x, valid_mask):
        """Execute one DINO-v1 Transformer block with padding-safe residuals."""
        attention_output = self._masked_dino_attention(block.attn, block.norm1(x), valid_mask)
        x = x + block.drop_path(attention_output)
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        mlp_output = block.mlp(block.norm2(x))
        mlp_output = self._zero_invalid_tokens(mlp_output, valid_mask)
        x = x + block.drop_path(mlp_output)
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x

    def _foreground_cls_batched(self, patch_tokens, patch_mask):
        foreground_tokens, foreground_valid_mask = self._pack_foreground_tokens(patch_tokens, patch_mask)
        batch_size = patch_tokens.shape[0]
        foreground_cls = self._foreground_cls_with_position().expand(batch_size, -1, -1)
        x = torch.cat([foreground_cls, foreground_tokens], dim=1)
        valid_mask = torch.cat(
            [
                torch.ones(batch_size, 1, device=x.device, dtype=torch.bool),
                foreground_valid_mask,
            ],
            dim=1,
        )

        for block in self.backbone.blocks:
            x = self._masked_dino_block(block, x, valid_mask)
        x = self.backbone.norm(x)
        x = self._zero_invalid_tokens(x, valid_mask)
        return x[:, 0]

    def forward_features(self, images, patch_mask=None):
        global_cls = self.backbone(images)

        patch_tokens = self._prepare_patch_tokens(images)
        patch_mask = self._align_patch_mask(
            patch_mask,
            batch_size=images.shape[0],
            token_count=patch_tokens.shape[1],
            device=images.device,
        )

        foreground_cls = self._foreground_cls_batched(patch_tokens, patch_mask)

        return self.fusion(global_cls, foreground_cls)

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            images, patch_mask = inputs
        else:
            images, patch_mask = inputs, None

        fused_cls = self.forward_features(images, patch_mask=patch_mask)
        return self.head(fused_cls)
