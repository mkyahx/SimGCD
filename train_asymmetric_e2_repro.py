import runpy

import model
from directional_distill_loss import GlobalToForegroundDistillLoss


model.DistillLoss = GlobalToForegroundDistillLoss
runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")
