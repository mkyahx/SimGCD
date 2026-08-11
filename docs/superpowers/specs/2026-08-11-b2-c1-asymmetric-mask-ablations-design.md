# B2 and C1 asymmetric-mask ablations

## Goal

Add isolated SimGCD ablation entrypoints on top of `simgcd-work`.  The existing asymmetric-mask implementation, its launchers, and the A2 control remain unchanged.

## Shared constraints

- Every new experiment uses `max_foreground_tokens=128` on CUB, Stanford Cars, and FGVC-Aircraft.
- The existing paired mask dataset, deterministic seed setup, SimGCD head, loss functions, and evaluation protocol are reused unchanged.
- Each ablation gets a dedicated model module, training entrypoint, and one Slurm submit script per benchmark.
- The implementation is additive only; it must not modify existing training or submit files.

## B2: foreground-to-foreground views

Both augmented image views use their own aligned TokenCut masks.  Each view is encoded through the existing foreground-token pipeline, including packed variable-length tokens and padding-safe attention.  Training returns the two foreground CLS features in the unchanged SimGCD ordering:

`[foreground(view 1); foreground(view 2)]`.

Evaluation uses the same foreground-only encoding on one image and its mask.  This changes only the view information assignment relative to the current global-to-foreground asymmetric baseline.

## C1: frozen shared DINO CLS initialization

C1 keeps the current global-to-foreground view assignment.  Its foreground branch prepends a frozen copy of the pretrained DINO `cls_token`, rather than the current new learnable foreground CLS parameter.  The copied token is not an `nn.Parameter` and receives no gradients.  Positional embedding handling, selected patch tokens, DINO blocks, trainable final backbone block, and head are otherwise unchanged.

## Files and launchers

New modules and thin `runpy` training wrappers will make the ablation-specific substitution before invoking the existing reproducible trainer.  Six submit scripts will use the benchmark presets already used by the asymmetric-mask launchers except for the common `max_foreground_tokens=128`:

- CUB: `memax_weight=2`
- Stanford Cars: `memax_weight=1`
- FGVC-Aircraft: `memax_weight=1`

All scripts run seeds `0 1 2` and identify their method and benchmark in job/output/experiment names.

## Verification

Tests will be written before implementation.  Source-contract tests will verify B2 forwards both masks and produces two foreground features, and C1 creates a non-parameter frozen CLS copy while preserving its position addition.  Script tests will check entrypoints, dataset names, masks root, and `K=128`.  The existing static regression suite and Python compilation will be run before commit and push.
