# B2 and C1 Asymmetric-Mask Ablations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add additive, reproducible B2 and C1 SimGCD ablation entrypoints and launchers on `simgcd-work`.

**Architecture:** B2 subclasses the existing asymmetric model and changes only training-time view routing to foreground-to-foreground.  Its trainer is an additive copy of the reproducible trainer because it must feed both masks. C1 subclasses the foreground encoder and replaces the trainable foreground CLS parameter with a frozen DINO CLS buffer; its thin entrypoint monkey-patches the existing reproducible trainer because its input interface is unchanged.

**Tech Stack:** Python, PyTorch, unittest source-contract tests, Slurm/bash, Git.

## Global Constraints

- Every new experiment uses `max_foreground_tokens=128` on CUB, Stanford Cars, and FGVC-Aircraft.
- Existing training, model, A2, and submit files remain unchanged.
- Each ablation gets a dedicated model module, `train_*_repro.py` wrapper, and three submit scripts.
- All scripts use the existing TokenCut mask root and seeds `0 1 2`.

---

### Task 1: B2 foreground-to-foreground model and entrypoint

**Files:**
- Create: `asymmetric_b2_mask_model.py`
- Create: `train_asymmetric_b2_repro.py`
- Modify: `tests/test_asymmetric_b2_c1_controls.py`

**Interfaces:**
- Consumes: `AsymmetricMaskModel._foreground_features(images, patch_mask)` and the existing `head`.
- Produces: `B2ForegroundPairMaskModel.forward((view_one, view_two, mask_one, mask_two))`, returning head outputs ordered as `[foreground(view 1); foreground(view 2)]`.

- [ ] **Step 1: Write the failing test**

```python
def test_b2_model_routes_both_training_views_through_foreground_encoding(self):
    source = (REPO_ROOT / "asymmetric_b2_mask_model.py").read_text(encoding="utf-8")
    self.assertIn("foreground_one = self._foreground_features(view_one, mask_one)", source)
    self.assertIn("foreground_two = self._foreground_features(view_two, mask_two)", source)
    self.assertIn("torch.cat([foreground_one, foreground_two], dim=0)", source)

def test_b2_trainer_imports_the_b2_model(self):
    source = (REPO_ROOT / "train_asymmetric_b2_repro.py").read_text(encoding="utf-8")
    self.assertIn("from asymmetric_b2_mask_model import B2ForegroundPairMaskModel", source)
    self.assertIn("model = B2ForegroundPairMaskModel(", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asymmetric_b2_c1_controls.AsymmetricB2C1ControlTests.test_b2_model_routes_both_training_views_through_foreground_encoding tests.test_asymmetric_b2_c1_controls.AsymmetricB2C1ControlTests.test_b2_entrypoint_patches_only_the_model_class -v`

Expected: FAIL because the B2 files do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
class B2ForegroundPairMaskModel(AsymmetricMaskModel):
    def forward(self, inputs):
        if self.training:
            view_one, view_two, mask_one, mask_two = inputs
            foreground_one = self._foreground_features(view_one, mask_one)
            foreground_two = self._foreground_features(view_two, mask_two)
            return self.head(torch.cat([foreground_one, foreground_two], dim=0))
        images, patch_mask = inputs
        return self.head(self._foreground_features(images, patch_mask))
```

The B2 trainer is an additive copy of `train_asymmetric_mask_repro.py`. It imports this class and instantiates `B2ForegroundPairMaskModel` in place of `AsymmetricMaskModel`; all other training setup is preserved.

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2.

Expected: PASS.

### Task 2: B2 paired-mask trainer adapter and launchers

**Files:**
- Create: `train_asymmetric_b2_repro.py` (extend Task 1 wrapper)
- Create: `scripts/submit_asymmetric_b2_cub_repro.sh`
- Create: `scripts/submit_asymmetric_b2_cars_repro.sh`
- Create: `scripts/submit_asymmetric_b2_aircraft_repro.sh`
- Modify: `tests/test_asymmetric_b2_c1_controls.py`

**Interfaces:**
- Consumes: existing `collate_train_mask_batch`, which returns two image views and two patch-mask views.
- Produces: a wrapper-local `train` that calls the B2 model with `(images[0], images[1], patch_mask[0], patch_mask[1])`; submit scripts call the B2 entrypoint with `--max_foreground_tokens 128`.

- [ ] **Step 1: Write the failing test**

```python
def test_b2_entrypoint_forwards_both_patch_masks(self):
    source = (REPO_ROOT / "train_asymmetric_b2_repro.py").read_text(encoding="utf-8")
    self.assertIn("student((images[0], images[1], patch_mask[0], patch_mask[1]))", source)

def test_b2_submit_scripts_use_the_benchmark_presets_and_k_128(self):
    expected = {"cub": "cub", "cars": "scars", "aircraft": "aircraft"}
    for benchmark, dataset in expected.items():
        source = (REPO_ROOT / "scripts" / f"submit_asymmetric_b2_{benchmark}_repro.sh").read_text(encoding="utf-8")
        self.assertIn("python train_asymmetric_b2_repro.py", source)
        self.assertIn(f"--dataset_name '{dataset}'", source)
        self.assertIn("--max_foreground_tokens 128", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asymmetric_b2_c1_controls.AsymmetricB2C1ControlTests.test_b2_entrypoint_forwards_both_patch_masks tests.test_asymmetric_b2_c1_controls.AsymmetricB2C1ControlTests.test_b2_submit_scripts_use_the_benchmark_presets_and_k_128 -v`

Expected: FAIL because the paired-mask wrapper path and scripts do not exist.

- [ ] **Step 3: Write minimal implementation**

Copy the existing reproducible trainer into the B2-specific file only when needed to replace the single call inside `train` with:

```python
student_proj, student_out = student((images[0], images[1], patch_mask[0], patch_mask[1]))
```

Add scripts that keep the existing benchmark-specific arguments and substitute only job identifiers, entrypoint, experiment names, and `--max_foreground_tokens 128`.

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2.

Expected: PASS.

### Task 3: C1 frozen DINO CLS model and entrypoint

**Files:**
- Create: `asymmetric_c1_mask_model.py`
- Create: `train_asymmetric_c1_repro.py`
- Modify: `tests/test_asymmetric_b2_c1_controls.py`

**Interfaces:**
- Consumes: `MaskForegroundGLSimModel` and `AsymmetricMaskModel`.
- Produces: `FrozenDinoClsMaskForegroundEncoder`, whose `foreground_cls_token` is a registered buffer cloned from DINO CLS, and `C1FrozenClsAsymmetricMaskModel` using it in the original global-to-foreground forward path.

- [ ] **Step 1: Write the failing test**

```python
def test_c1_uses_a_frozen_dino_cls_buffer_not_a_parameter(self):
    source = (REPO_ROOT / "asymmetric_c1_mask_model.py").read_text(encoding="utf-8")
    self.assertIn("self.register_buffer(\"foreground_cls_token\"", source)
    self.assertIn("backbone.cls_token.detach().clone()", source)
    self.assertNotIn("nn.Parameter", source)
    self.assertIn("foreground_cls = self.foreground_cls_token", source)

def test_c1_entrypoint_patches_only_the_model_class(self):
    source = (REPO_ROOT / "train_asymmetric_c1_repro.py").read_text(encoding="utf-8")
    self.assertIn("asymmetric_mask_model.AsymmetricMaskModel = C1FrozenClsAsymmetricMaskModel", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asymmetric_b2_c1_controls.AsymmetricB2C1ControlTests.test_c1_uses_a_frozen_dino_cls_buffer_not_a_parameter tests.test_asymmetric_b2_c1_controls.AsymmetricB2C1ControlTests.test_c1_entrypoint_patches_only_the_model_class -v`

Expected: FAIL because the C1 files do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
class FrozenDinoClsMaskForegroundEncoder(MaskForegroundGLSimModel):
    def _init_foreground_cls(self):
        cls_token = self.backbone.cls_token.detach().clone()
        del self.foreground_cls_token
        self.register_buffer("foreground_cls_token", cls_token)
```

Construct this encoder inside `C1FrozenClsAsymmetricMaskModel`, freeze the unused fusion block as in the current asymmetric model, and preserve the existing global-to-foreground `forward` implementation.  The thin entrypoint monkey-patches only `AsymmetricMaskModel` before calling the existing trainer.

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2.

Expected: PASS.

### Task 4: C1 launchers and full regression verification

**Files:**
- Create: `scripts/submit_asymmetric_c1_cub_repro.sh`
- Create: `scripts/submit_asymmetric_c1_cars_repro.sh`
- Create: `scripts/submit_asymmetric_c1_aircraft_repro.sh`
- Modify: `tests/test_asymmetric_b2_c1_controls.py`

**Interfaces:**
- Consumes: `train_asymmetric_c1_repro.py`.
- Produces: three zero-argument Slurm launchers that select the C1 entrypoint and `K=128`.

- [ ] **Step 1: Write the failing test**

```python
def test_c1_submit_scripts_use_the_benchmark_presets_and_k_128(self):
    expected = {"cub": "cub", "cars": "scars", "aircraft": "aircraft"}
    for benchmark, dataset in expected.items():
        source = (REPO_ROOT / "scripts" / f"submit_asymmetric_c1_{benchmark}_repro.sh").read_text(encoding="utf-8")
        self.assertIn("python train_asymmetric_c1_repro.py", source)
        self.assertIn(f"--dataset_name '{dataset}'", source)
        self.assertIn("--max_foreground_tokens 128", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asymmetric_b2_c1_controls.AsymmetricB2C1ControlTests.test_c1_submit_scripts_use_the_benchmark_presets_and_k_128 -v`

Expected: FAIL because the C1 scripts do not exist.

- [ ] **Step 3: Write minimal implementation**

Add one C1 script per benchmark, matching the corresponding existing asymmetric script except for C1 identifiers, C1 entrypoint, C1 experiment names, and `--max_foreground_tokens 128`.

- [ ] **Step 4: Run all verification**

Run:

```powershell
python -m unittest tests.test_asymmetric_b2_c1_controls tests.test_asymmetric_a2_control tests.test_asymmetric_all_ones_control tests.test_asymmetric_mask_submit_scripts tests.test_glsim_mask_source_contract -v
python -m py_compile asymmetric_b2_mask_model.py train_asymmetric_b2_repro.py asymmetric_c1_mask_model.py train_asymmetric_c1_repro.py
git diff --check
```

Expected: all static tests pass, compilation succeeds, and the diff check is clean.

- [ ] **Step 5: Commit and push**

```powershell
git add asymmetric_b2_mask_model.py train_asymmetric_b2_repro.py asymmetric_c1_mask_model.py train_asymmetric_c1_repro.py scripts/submit_asymmetric_b2_cub_repro.sh scripts/submit_asymmetric_b2_cars_repro.sh scripts/submit_asymmetric_b2_aircraft_repro.sh scripts/submit_asymmetric_c1_cub_repro.sh scripts/submit_asymmetric_c1_cars_repro.sh scripts/submit_asymmetric_c1_aircraft_repro.sh tests/test_asymmetric_b2_c1_controls.py docs/superpowers/plans/2026-08-11-b2-c1-asymmetric-mask-ablations.md
git commit -m "Add B2 and C1 mask ablation launchers"
git push origin simgcd-work
```
