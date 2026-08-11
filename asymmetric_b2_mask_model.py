import torch

from asymmetric_mask_model import AsymmetricMaskModel


class B2ForegroundPairMaskModel(AsymmetricMaskModel):
    """B2 control: align foreground-only representations across both views."""

    def forward(self, inputs):
        if self.training:
            view_one, view_two, mask_one, mask_two = inputs
            foreground_one = self._foreground_features(view_one, mask_one)
            foreground_two = self._foreground_features(view_two, mask_two)
            return self.head(torch.cat([foreground_one, foreground_two], dim=0))

        images, patch_mask = inputs
        return self.head(self._foreground_features(images, patch_mask))
