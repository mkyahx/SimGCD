"""A2 control: random token locations with TokenCut-matched cardinality."""

import torch

from asymmetric_mask_model import AsymmetricMaskModel


class A2CardinalityMatchedRandomMaskModel(AsymmetricMaskModel):
    """Replace TokenCut locations with random patches while preserving token counts.

    The foreground view receives the same number of usable tokens as the TokenCut
    branch would receive after its ``max_foreground_tokens`` cap.  The original
    mask is used only to count tokens, never to choose their spatial positions.
    """

    def _foreground_features(self, images, patch_mask):
        patch_tokens = self.mask_encoder._prepare_patch_tokens(images)
        aligned_mask = self.mask_encoder._align_patch_mask(
            patch_mask,
            batch_size=images.shape[0],
            token_count=patch_tokens.shape[1],
            device=images.device,
        )
        batch_size, token_count = aligned_mask.shape
        random_mask = torch.zeros_like(aligned_mask)

        for batch_index in range(batch_size):
            foreground_count = int(aligned_mask[batch_index].sum().item())
            if foreground_count < self.mask_encoder.min_foreground_tokens:
                target_count = token_count
            elif self.mask_encoder.max_foreground_tokens is None:
                target_count = foreground_count
            else:
                target_count = min(foreground_count, self.mask_encoder.max_foreground_tokens)

            selected_indices = torch.randperm(token_count, device=patch_tokens.device)[:target_count]
            random_mask[batch_index, selected_indices] = True

        return self.mask_encoder._foreground_cls_batched(patch_tokens, random_mask)
