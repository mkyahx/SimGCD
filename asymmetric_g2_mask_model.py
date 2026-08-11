from asymmetric_c1_mask_model import C1FrozenClsAsymmetricMaskModel


class GlobalOnlyInferenceAsymmetricMaskModel(C1FrozenClsAsymmetricMaskModel):
    """Keep asymmetric training but classify with the global CLS at evaluation."""

    def forward(self, inputs):
        if self.training:
            return super().forward(inputs)

        images, _ = inputs
        global_cls = self.mask_encoder.backbone(images)
        return self.head(global_cls)
