import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LASTViT1DChannelBackbone(nn.Module):
    """
    LaSt-ViT channel-domain 1D FFT backbone wrapper.

    This follows the original LaSt-ViT replacement idea: drop the encoded CLS
    token, filter every patch token along its channel dimension, select stable
    patch channels with Top-K over patches, and average selected original patch
    tokens into a CLS-like image feature.
    """

    def __init__(
        self,
        backbone: nn.Module,
        topk: int = 1,
        eps: float = 1e-6,
        token_source: str = "patch",
        sigma: float = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self._infer_embed_dim(backbone)
        self.topk = max(1, int(topk))
        self.eps = float(eps)
        self.token_source = token_source
        self.sigma = float(sigma) if sigma is not None else math.sqrt(float(self.embed_dim))
        self.cached_kernel = None
        self.cached_kernel_meta = None

    @staticmethod
    def _infer_embed_dim(backbone: nn.Module) -> int:
        for attr in ("embed_dim", "num_features", "hidden_size"):
            if hasattr(backbone, attr):
                return int(getattr(backbone, attr))
        if hasattr(backbone, "cls_token"):
            return int(backbone.cls_token.shape[-1])
        if hasattr(backbone, "pos_embed"):
            return int(backbone.pos_embed.shape[-1])
        raise RuntimeError("Could not infer backbone feature dimension.")

    def _patch_size(self):
        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(self.backbone, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            return int(patch_size[0]), int(patch_size[1])
        if patch_size is not None:
            patch_size = int(patch_size)
            return patch_size, patch_size
        return None

    def _infer_grid(self, num_patches: int, x: torch.Tensor):
        patch_size = self._patch_size()
        if patch_size is not None:
            ph, pw = patch_size
            gh = int(x.shape[-2]) // ph
            gw = int(x.shape[-1]) // pw
            if gh * gw == num_patches:
                return gh, gw

        side = int(math.sqrt(num_patches))
        if side * side == num_patches:
            return side, side
        raise ValueError(f"Cannot infer patch grid for {num_patches} tokens.")

    def masks_to_patch_mask(self, masks: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        if masks is None:
            return None
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if masks.ndim != 4:
            raise ValueError(f"Expected masks with shape [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")
        patch_mask = F.interpolate(masks.float(), size=(grid_h, grid_w), mode="nearest")
        patch_mask = patch_mask.flatten(2).squeeze(1) > 0.5
        empty = ~patch_mask.any(dim=1)
        if empty.any():
            patch_mask[empty] = True
        return patch_mask

    def _prepare_tokens_fallback(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "prepare_tokens"):
            return self.backbone.prepare_tokens(x)
        if hasattr(self.backbone, "prepare_tokens_with_masks"):
            return self.backbone.prepare_tokens_with_masks(x, None)
        if not hasattr(self.backbone, "patch_embed"):
            raise RuntimeError("Backbone does not expose forward_features, prepare_tokens, or patch_embed.")

        width, height = int(x.shape[-1]), int(x.shape[-2])
        patch_tokens = self.backbone.patch_embed(x)
        if patch_tokens.dim() == 4:
            patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        cls_tokens = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        if hasattr(self.backbone, "interpolate_pos_encoding"):
            tokens = tokens + self.backbone.interpolate_pos_encoding(tokens, width, height)
        else:
            tokens = tokens + self.backbone.pos_embed
        if hasattr(self.backbone, "pos_drop"):
            tokens = self.backbone.pos_drop(tokens)
        return tokens

    def _features_from_dict(self, features):
        if self.token_source == "all" and "x_prenorm" in features:
            tokens = features["x_prenorm"]
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                tokens = norm(tokens)
            num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
            return tokens[:, num_register_tokens + 1 :]

        for key in ("x_norm_patchtokens", "patch_tokens", "patchtokens"):
            if key in features:
                return features[key]

        for key in ("last_hidden_state", "x_norm"):
            if key in features:
                tokens = features[key]
                num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
                return tokens[:, num_register_tokens + 1 :]

        raise RuntimeError("forward_features() did not return patch tokens.")

    def forward_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
            if isinstance(features, dict):
                return self._features_from_dict(features)
            if isinstance(features, (tuple, list)):
                for item in features:
                    if isinstance(item, dict):
                        return self._features_from_dict(item)
                for item in features:
                    if torch.is_tensor(item) and item.ndim == 3:
                        return item[:, 1:]
            if torch.is_tensor(features) and features.ndim == 3:
                return features[:, 1:]

        tokens = self._prepare_tokens_fallback(x)
        for block in getattr(self.backbone, "blocks"):
            tokens = block(tokens)
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            tokens = norm(tokens)
        num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
        return tokens[:, num_register_tokens + 1 :]

    def gaussian_kernel_1d(self, kernel_size: int, device, dtype) -> torch.Tensor:
        offsets = torch.arange(
            -kernel_size // 2 + 1,
            kernel_size // 2 + 1,
            device=device,
            dtype=dtype,
        )
        sigma = max(self.sigma, self.eps)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / torch.max(kernel).clamp_min(self.eps)
        return kernel.view(1, 1, kernel_size)

    def get_gaussian_kernel(self, feat_dim: int, device, dtype) -> torch.Tensor:
        meta = (feat_dim, float(self.sigma), device, dtype)
        if self.cached_kernel is None or self.cached_kernel_meta != meta:
            self.cached_kernel = self.gaussian_kernel_1d(feat_dim, device, dtype)
            self.cached_kernel_meta = meta
        return self.cached_kernel

    def channel_frequency_filter(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens_float = patch_tokens.float()
        feat_dim = patch_tokens_float.shape[-1]
        kernel = self.get_gaussian_kernel(feat_dim, patch_tokens_float.device, patch_tokens_float.dtype)

        spectrum = torch.fft.fft(patch_tokens_float, dim=-1)
        spectrum = torch.fft.fftshift(spectrum, dim=-1)
        filtered_spectrum = spectrum * kernel
        filtered_spectrum = torch.fft.ifftshift(filtered_spectrum, dim=-1)
        return torch.fft.ifft(filtered_spectrum, dim=-1).real

    def stability_scores(self, patch_tokens: torch.Tensor, filtered_tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens_float = patch_tokens.float()
        denom = torch.abs(filtered_tokens - patch_tokens_float).clamp_min(self.eps)
        return patch_tokens_float / denom

    def channel_topk_select(self, patch_tokens: torch.Tensor, scores: torch.Tensor, patch_mask: torch.Tensor = None) -> torch.Tensor:
        if patch_mask is not None:
            outputs = []
            for sample_tokens, sample_scores, sample_mask in zip(patch_tokens, scores, patch_mask):
                valid_tokens = sample_tokens[sample_mask]
                valid_scores = sample_scores[sample_mask]
                if valid_tokens.shape[0] == 0:
                    valid_tokens = sample_tokens
                    valid_scores = sample_scores
                outputs.append(self.channel_topk_select(valid_tokens.unsqueeze(0), valid_scores.unsqueeze(0), None).squeeze(0))
            return torch.stack(outputs, dim=0)

        patch_k = min(self.topk, patch_tokens.shape[1])
        _, indices = torch.topk(scores, k=patch_k, dim=1, largest=True)
        selected = torch.gather(patch_tokens, 1, indices)
        return selected.mean(dim=1)

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        patch_tokens = self.forward_patch_tokens(x)
        filtered_tokens = self.channel_frequency_filter(patch_tokens)
        scores = self.stability_scores(patch_tokens, filtered_tokens)
        patch_mask = None
        if masks is not None:
            grid_h, grid_w = self._infer_grid(patch_tokens.shape[1], x)
            patch_mask = self.masks_to_patch_mask(masks, grid_h, grid_w)
        return self.channel_topk_select(patch_tokens, scores, patch_mask)
