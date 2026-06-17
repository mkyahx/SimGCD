import math

import torch
import torch.nn as nn


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

    def channel_topk_select(self, patch_tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        patch_k = min(self.topk, patch_tokens.shape[1])
        _, indices = torch.topk(scores, k=patch_k, dim=1, largest=True)
        selected = torch.gather(patch_tokens, 1, indices)
        return selected.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.forward_patch_tokens(x)
        filtered_tokens = self.channel_frequency_filter(patch_tokens)
        scores = self.stability_scores(patch_tokens, filtered_tokens)
        return self.channel_topk_select(patch_tokens, scores)
