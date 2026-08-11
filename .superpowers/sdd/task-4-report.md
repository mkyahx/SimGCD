# Task 4 report

## Scope

- Added C1 Slurm launchers for CUB, Stanford Cars, and FGVC-Aircraft.
- Added the C1 launcher source-contract test.
- Did not modify existing training/model code, B2/A2 code, or all-ones files.

## TDD evidence

The new C1 launcher test was added first and executed before the scripts existed.
It failed as expected with:

```text
FileNotFoundError: ...\\scripts\\submit_asymmetric_c1_cub_repro.sh
```

## Implementation

- `scripts/submit_asymmetric_c1_cub_repro.sh`
- `scripts/submit_asymmetric_c1_cars_repro.sh`
- `scripts/submit_asymmetric_c1_aircraft_repro.sh`

Each launcher calls `train_asymmetric_c1_repro.py`, preserves the corresponding
baseline resource and optimization preset, retains seeds `0 1 2` and the mask
root, and sets `--max_foreground_tokens 128`.

## Verification

Command:

```powershell
& 'C:\\Users\\张申如\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' -m unittest tests.test_asymmetric_b2_c1_controls tests.test_asymmetric_a2_control tests.test_asymmetric_all_ones_control tests.test_asymmetric_mask_submit_scripts tests.test_glsim_mask_source_contract -v
```

Result: 20 tests passed.

Also passed:

```powershell
& 'C:\\Users\\张申如\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' -m py_compile asymmetric_b2_mask_model.py train_asymmetric_b2_repro.py asymmetric_c1_mask_model.py train_asymmetric_c1_repro.py
git diff --check
```

## Concerns

- The local runtime does not include PyTorch/CUDA, so no actual Slurm or training
  execution was performed; verification is static/source-contract based.
- The requested all-ones test module was pre-existing and passed unchanged.
