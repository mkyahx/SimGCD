# Task 1 Report — B2 Foreground Pair Model and Entrypoint

## Status

Completed in isolated worktree `C:\tmp\simgcd-b2-task1` on branch `codex/task-b2-model`, based on `simgcd-work` commit `40588a0`.

## Changed files

- `asymmetric_b2_mask_model.py`: adds `B2ForegroundPairMaskModel`, which encodes both training views through `_foreground_features` using their corresponding masks, then returns head outputs ordered as `[foreground(view 1); foreground(view 2)]`.
- `train_asymmetric_b2_repro.py`: additive B2 trainer. It retains the asymmetric-mask data/setup path and calls the B2 model with `(images[0], images[1], patch_mask[0], patch_mask[1])`.
- `tests/test_asymmetric_b2_c1_controls.py`: source-contract coverage for B2 model routing and B2 trainer import/instantiation.

## TDD evidence

1. RED: the required two-test command failed with `FileNotFoundError` for the missing B2 model and trainer files.
2. GREEN: after the implementation, the same command passed: `Ran 2 tests ... OK`.

## Verification

- Static source-contract tests: passed (2/2).
- `git diff --check`: passed.
- Runtime PyTorch execution was not performed because the available local Python runtime does not provide a usable Torch/GPU environment. A first `py_compile` attempt also could not create `__pycache__` in the isolated worktree due to Windows permission restrictions.

## Scope

No Task 2 submission scripts were created and no all-ones files were modified.
