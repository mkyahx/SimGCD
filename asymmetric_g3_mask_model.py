from asymmetric_mask_model import AsymmetricMaskModel


class MeanFeatureInferenceAsymmetricMaskModel(AsymmetricMaskModel):
    """Keep asymmetric training and average global/foreground CLS before the head."""

    def forward(self, inputs):
        if self.training:
            return super().forward(inputs)

        images, patch_mask = inputs
        global_cls = self.mask_encoder.backbone(images)
        foreground_cls = self._foreground_features(images, patch_mask)
        fused_cls = 0.5 * (global_cls + foreground_cls)
        return self.head(fused_cls)
