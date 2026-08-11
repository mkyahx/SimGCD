import runpy

import asymmetric_mask_model
from asymmetric_c1_mask_model import C1FrozenClsAsymmetricMaskModel


asymmetric_mask_model.AsymmetricMaskModel = C1FrozenClsAsymmetricMaskModel
runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")
