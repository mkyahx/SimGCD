import runpy

import asymmetric_mask_model
from asymmetric_h2_mask_model import H2BackgroundMaskAsymmetricModel


asymmetric_mask_model.AsymmetricMaskModel = H2BackgroundMaskAsymmetricModel
runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")
