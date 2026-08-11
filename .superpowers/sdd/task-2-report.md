# Task 2 Report: B2 Paired-Mask Launchers and Contracts

## Scope

- Added the B2 paired-mask trainer source-contract assertion.
- Added three benchmark-specific B2 Slurm launchers:
  - `scripts/submit_asymmetric_b2_cub_repro.sh`
  - `scripts/submit_asymmetric_b2_cars_repro.sh`
  - `scripts/submit_asymmetric_b2_aircraft_repro.sh`
- Kept the existing trainer/model files unchanged because the parent Task 1 commit already forwards both augmented patch masks.

## TDD Evidence

1. Added the two Task 2 contract tests in `tests/test_asymmetric_b2_c1_controls.py`.
2. Ran the specified tests before creating launchers. The four-mask trainer assertion passed; the launcher test failed with `FileNotFoundError` for `submit_asymmetric_b2_cub_repro.sh`, as expected.
3. Added the three launchers based on the corresponding asymmetric-mask presets.
4. Re-ran the complete B2/C1 contract test module: 4 tests passed.

## Presets Preserved

- CUB: original `memax_weight=2`, six-hour allocation, seeds `0 1 2`; only B2 identifiers/entrypoint and `max_foreground_tokens=128` differ from the source script.
- Cars: original `memax_weight=1`, eighteen-hour allocation, dataset name `scars`, seeds `0 1 2`; B2 identifiers/entrypoint and `max_foreground_tokens=128` used.
- Aircraft: original `memax_weight=1`, eighteen-hour allocation, seeds `0 1 2`; B2 identifiers/entrypoint and `max_foreground_tokens=128` used.

## Verification

```text
python -m unittest tests.test_asymmetric_b2_c1_controls -v

Ran 4 tests in 0.008s
OK
```

`git diff --check` returned exit code 0.

## Concerns

- The local environment has no configured Slurm cluster or PyTorch/CUDA runtime, so the new launchers were verified statically rather than submitted or trained.
