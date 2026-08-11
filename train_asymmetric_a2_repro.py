"""Run the A2 cardinality-matched random-token control without changing the base trainer."""

import runpy

import asymmetric_mask_model
from asymmetric_a2_mask_model import A2CardinalityMatchedRandomMaskModel


# The original trainer imports this symbol at startup; replace only that model
# class so its data pipeline, losses, optimizer, seeds, and launch contract stay
# identical to the TokenCut asymmetric run.
asymmetric_mask_model.AsymmetricMaskModel = A2CardinalityMatchedRandomMaskModel


if __name__ == "__main__":
    runpy.run_module("train_asymmetric_mask_repro", run_name="__main__")
