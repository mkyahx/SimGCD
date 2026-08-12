from asymmetric_mask_model import AsymmetricMaskModel


class H2BackgroundMaskAsymmetricModel(AsymmetricMaskModel):
    """Current asymmetric baseline with TokenCut's complementary background tokens.

    The first view remains global.  The second view uses the complement of the
    aligned TokenCut foreground mask and otherwise keeps the current asymmetric
    model, including its learnable foreground CLS token, unchanged.
    """

    def _foreground_features(self, images, patch_mask):
        patch_tokens = self.mask_encoder._prepare_patch_tokens(images)
        foreground_patch_mask = self.mask_encoder._align_patch_mask(
            patch_mask,
            batch_size=images.shape[0],
            token_count=patch_tokens.shape[1],
            device=images.device,
        )
        background_patch_mask = ~foreground_patch_mask
        return self.mask_encoder._foreground_cls_batched(patch_tokens, background_patch_mask)
