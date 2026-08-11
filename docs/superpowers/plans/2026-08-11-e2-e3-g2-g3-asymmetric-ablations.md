# E2/E3/G2/G3 Asymmetric Ablations Implementation Plan

**Goal:** Add directional-alignment (E2/E3) and inference-readout (G2/G3) controls on `simgcd-work` without changing existing training files.

**Architecture:** E2 and E3 add a directional replacement for SimGCD's bidirectional `DistillLoss` and inject it through additive entrypoints before executing `train_asymmetric_mask_repro`. E2 uses detached global(view 1) probabilities as the teacher for foreground(view 2); E3 reverses this. G2 and G3 add model subclasses that keep global→foreground training unchanged but return global-only or pre-head mean(global, foreground) features during evaluation.

## Constraints

- All implementation files and submit scripts are new; existing asymmetric, B2, C1, A2, and all-ones files remain unchanged.
- Every CUB, Cars, and Aircraft launcher uses `--max_foreground_tokens 128` and preserves its existing benchmark preset.
- E2/E3 change only the cross-view cluster alignment direction; all other losses, data loading, seed setup, and evaluation code stay unchanged.
- G2/G3 change only evaluation feature selection; training remains `[global(view 1); foreground(view 2)]`.

## Test-first tasks

1. Add failing source-contract tests for the directional loss and E2/E3 monkey-patch entrypoints; implement the loss modules/entrypoints and verify.
2. Add failing script-contract tests; add six E2/E3 submit scripts and verify `K=128`, dataset IDs, masks root, seeds, and per-benchmark `memax_weight`.
3. Add failing source-contract tests for G2 global-only and G3 averaged inference; implement subclasses and entrypoints, preserving inherited training forward behavior.
4. Add failing script-contract tests; add six G2/G3 submit scripts and run the full static regression suite, compilation, and diff check.
