"""Run the current asymmetric model with an all-ones spatial-mask control."""

import runpy
import sys

import numpy as np

from data.mask_dataset import DatasetWithPatchMask


def _all_ones_mask(self, idx):
    """Select the complete 14x14 ViT patch grid without reading TokenCut masks."""
    return np.ones((14, 14), dtype=np.float32)


DatasetWithPatchMask._get_mask = _all_ones_mask

# The original parser requires this argument, but the patched dataset never
# dereferences it.
if "--mask_root" not in sys.argv:
    sys.argv.extend(["--mask_root", "__all_ones_control__"])


if __name__ == "__main__":
    runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")
