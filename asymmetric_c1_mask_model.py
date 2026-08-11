import torch
import torch.nn as nn

from glsim_mask_model import MaskForegroundGLSimModel


class FrozenDinoClsMaskForegroundEncoder(MaskForegroundGLSimModel):
    """Foreground encoder whose prepended CLS is a frozen DINO buffer."""

    def _init_foreground_cls(self):
        cls_token = self.backbone.cls_token.detach().clone()
        del self.foreground_cls_token
        self.register_buffer("foreground_cls_token", cls_token)
        # The inherited _foreground_cls_with_position starts with:
        # foreground_cls = self.foreground_cls_token


class C1FrozenClsAsymmetricMaskModel(nn.Module):
    """Original global-to-foreground asymmetric model with a frozen foreground CLS."""

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
        self.mask_encoder = FrozenDinoClsMaskForegroundEncoder(
            backbone=backbone,
            head=nn.Identity(),
            feat_dim=feat_dim,
            fusion_heads=fusion_heads,
            min_foreground_tokens=min_foreground_tokens,
            max_foreground_tokens=max_foreground_tokens,
        )
        self.head = head

        # This experiment does not use GLSim's dual-CLS fusion block.
        for parameter in self.mask_encoder.fusion.parameters():
            parameter.requires_grad = False

    def _foreground_features(self, images, patch_mask):
        patch_tokens = self.mask_encoder._prepare_patch_tokens(images)
        patch_mask = self.mask_encoder._align_patch_mask(
            patch_mask,
            batch_size=images.shape[0],
            token_count=patch_tokens.shape[1],
            device=images.device,
        )
        return self.mask_encoder._foreground_cls_batched(patch_tokens, patch_mask)

    def forward(self, inputs):
        if self.training:
            global_images, foreground_images, foreground_patch_mask = inputs
            global_cls = self.mask_encoder.backbone(global_images)
            foreground_cls = self._foreground_features(foreground_images, foreground_patch_mask)
            return self.head(torch.cat([global_cls, foreground_cls], dim=0))

        images, patch_mask = inputs
        foreground_cls = self._foreground_features(images, patch_mask)
        return self.head(foreground_cls)
