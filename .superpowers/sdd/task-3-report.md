# Task 3 report

## Status

Implemented C1 frozen-DINO-CLS ablation as additive files only.

## Changes

- Added `asymmetric_c1_mask_model.py`.
  - `FrozenDinoClsMaskForegroundEncoder` replaces the inherited trainable
    foreground CLS parameter with a buffer cloned from
    `backbone.cls_token.detach().clone()`.
  - `C1FrozenClsAsymmetricMaskModel` preserves the existing
    global-view-to-foreground-view training order and foreground-only inference.
- Added `train_asymmetric_c1_repro.py`.
  - Monkey-patches only `AsymmetricMaskModel` before dispatching to the original
    asymmetric trainer.
- Extended `tests/test_asymmetric_b2_c1_controls.py` with C1 source-contract
  coverage.

## Verification

- RED: the two C1 tests failed before the C1 files existed.
- GREEN: the two C1 tests pass after implementation.
- `py_compile` passes for both new Python files.
- Runtime training was not run because the local bundled Python environment has
  no PyTorch/CUDA installation.

## Buffer behavior

`register_buffer` includes the frozen CLS in `state_dict`, moves it with
`.to(device)`, and excludes it from `model.parameters()`; therefore the existing
`get_params_groups` path naturally does not optimize it.
