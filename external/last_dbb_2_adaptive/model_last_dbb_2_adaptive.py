import math

import torch
import torch.nn as nn


class LASTDBB2AdaptiveBackbone(nn.Module):
    """
    2D spatial-frequency LAST with fixed/adaptive Gaussian masks.

    Keeps the vote-based patch selection from model_last_dbb_2.py. The score is
    always stable: abs(original) / abs(filtered - original).
    """

    def __init__(
        self,
        backbone: nn.Module,
        topk: int = 4,
        vote_topk: int = None,
        eps: float = 1e-6,
        token_source: str = "patch",
        mask_mode: str = "fixed",
        sigma: float = None,
        adaptive_k: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self._infer_embed_dim(backbone)
        self.topk = max(1, int(topk))
        self.vote_topk = max(1, int(vote_topk)) if vote_topk is not None else self.topk
        self.eps = float(eps)
        self.token_source = token_source
        self.mask_mode = mask_mode
        self.sigma = float(sigma) if sigma is not None else math.sqrt(float(self.embed_dim))
        self.adaptive_k = float(adaptive_k)

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

    def radial_grid(self, grid_h: int, grid_w: int, device, dtype):
        yy = torch.arange(grid_h, device=device, dtype=dtype) - (grid_h - 1) / 2.0
        xx = torch.arange(grid_w, device=device, dtype=dtype) - (grid_w - 1) / 2.0
        yy = yy.view(grid_h, 1)
        xx = xx.view(1, grid_w)
        radius = torch.sqrt(yy.pow(2) + xx.pow(2))
        max_radius = radius.max().clamp_min(self.eps)
        return radius / max_radius * (0.5 * float(self.embed_dim))

    def fixed_gaussian_mask(self, radius: torch.Tensor) -> torch.Tensor:
        sigma = max(self.sigma, self.eps)
        return torch.exp(-radius.pow(2) / (2.0 * sigma ** 2)).view(1, 1, *radius.shape)

    def adaptive_gaussian_mask(self, amplitude: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        energy = amplitude.pow(2).mean(dim=1)
        denom = energy.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        radial_energy = (energy * radius.view(1, *radius.shape)).sum(dim=(-2, -1), keepdim=True) / denom
        sigma = (self.adaptive_k * radial_energy).clamp_min(self.eps)
        return torch.exp(-radius.view(1, *radius.shape).pow(2) / (2.0 * sigma.pow(2))).unsqueeze(1)

    def gaussian_mask(self, amplitude: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        if self.mask_mode == "adaptive":
            return self.adaptive_gaussian_mask(amplitude, radius)
        if self.mask_mode != "fixed":
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")
        return self.fixed_gaussian_mask(radius).to(dtype=amplitude.dtype)

    def spatial_frequency_filter(self, patch_tokens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, channels = patch_tokens.shape
        grid_h, grid_w = self._infer_grid(num_patches, x)
        fmap = patch_tokens.view(batch_size, grid_h, grid_w, channels).permute(0, 3, 1, 2)

        spectrum = torch.fft.fft2(fmap, dim=(-2, -1), norm="ortho")
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)

        radius = self.radial_grid(grid_h, grid_w, device=fmap.device, dtype=fmap.dtype)
        mask = self.gaussian_mask(amplitude, radius).to(dtype=amplitude.dtype)

        filtered_spectrum = torch.polar(amplitude * mask, phase)
        filtered_spectrum = torch.fft.ifftshift(filtered_spectrum, dim=(-2, -1))
        filtered = torch.fft.ifft2(filtered_spectrum, dim=(-2, -1), norm="ortho").real
        return filtered.permute(0, 2, 3, 1).reshape(batch_size, num_patches, channels)

    def stability_scores(self, patch_tokens: torch.Tensor, filtered_tokens: torch.Tensor) -> torch.Tensor:
        denom = torch.abs(filtered_tokens - patch_tokens).clamp_min(self.eps)
        return torch.abs(patch_tokens) / denom

    def vote_select(self, patch_tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        channel_k = min(self.vote_topk, patch_tokens.shape[1])
        _, channel_indices = torch.topk(scores, k=channel_k, dim=1, largest=True)

        votes = torch.zeros(scores.shape[0], scores.shape[1], device=scores.device, dtype=scores.dtype)
        ones = torch.ones_like(channel_indices, dtype=scores.dtype)
        votes.scatter_add_(1, channel_indices.reshape(scores.shape[0], -1), ones.reshape(scores.shape[0], -1))

        patch_k = min(self.topk, patch_tokens.shape[1])
        _, patch_indices = torch.topk(votes, k=patch_k, dim=1, largest=True)
        gather_indices = patch_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        selected = torch.gather(patch_tokens, 1, gather_indices)
        return selected.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.forward_patch_tokens(x)
        filtered_tokens = self.spatial_frequency_filter(patch_tokens, x)
        scores = self.stability_scores(patch_tokens, filtered_tokens)
        return self.vote_select(patch_tokens, scores)
