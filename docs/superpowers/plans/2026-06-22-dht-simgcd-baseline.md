# dHT–SimGCD Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SimGCD's fixed DINO v1 patch tokenizer with a vendored dHT tokenizer that emits a fixed sequence of 196 region slots and optionally allocates more real regions inside TokenCut masks than outside them.

**Architecture:** Keep DINO v1 ViT-B/16 transformer blocks, CLS token, normalization, SimGCD losses, data splits, and evaluation. A SimGCD adapter wraps vendored dHT, computes foreground/background budgets, produces region embeddings plus a padding mask, and feeds them through DINO blocks with masked attention.

**Tech Stack:** Python 3.10+, PyTorch 2.7+, torchvision 0.22+, pytest, DINO v1 ViT-B/16, vendored dHT.

## Global Constraints

- Work only on branch `feat/dht-simgcd-baseline`.
- Use `facebookresearch/dino:main` model `dino_vitb16`, feature dimension `768`.
- Always expose exactly `196` visual-token slots plus one CLS token.
- Padded slots are not valid regions and must be excluded by attention masks.
- Mask modes are exactly `none`, `fixed`, and `area_weighted`.
- Fixed mode uses `--dht_fg_ratio`; area-weighted mode uses `--dht_fg_density`.
- Empty, full, missing, or invalid masks fall back safely without deleting background information.
- Training and evaluation both load masks from `--mask_root`.
- SimGCD losses, prototypes, splits, and metrics remain unchanged.
- Vendored dHT retains its MIT license and exact source commit.
- SLURM execution must not require network access for dHT source code.
- Every production behavior is introduced with a failing test first.

---

### Task 1: Vendor and expose upstream dHT

**Files:**
- Create: `third_party/dht/dht/**`
- Create: `third_party/dht/LICENSE`
- Create: `third_party/dht/SOURCE.md`
- Create: `dht_simgcd/__init__.py`
- Test: `tests/test_dht_vendor.py`

**Interfaces:**
- Produces: importable `dht` package when `third_party/dht` is on `sys.path`.
- Produces: `dht_simgcd.ensure_vendored_dht_importable() -> pathlib.Path`.

- [ ] **Step 1: Write the failing vendor test**

```python
def test_vendored_dht_source_metadata_and_import():
    from dht_simgcd import ensure_vendored_dht_importable
    root = ensure_vendored_dht_importable()
    assert (root / "LICENSE").is_file()
    assert (root / "SOURCE.md").read_text().find("c825dde939302aa8c026802442ddbfc2028cc8e5") >= 0
    from dht.tok.tokenizer import dHTTokenizer
    assert dHTTokenizer.__name__ == "dHTTokenizer"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_dht_vendor.py -v`

Expected: import failure because `dht_simgcd` and vendored files do not exist.

- [ ] **Step 3: Copy upstream package and metadata**

Copy `dHT/dht` without modification into `third_party/dht/dht`, copy the MIT
license, and write `SOURCE.md` with repository URL, source commit
`c825dde939302aa8c026802442ddbfc2028cc8e5`, copy date, and statement that
SimGCD-specific adapters live outside the vendored tree.

- [ ] **Step 4: Add the import helper**

```python
from pathlib import Path
import sys

def ensure_vendored_dht_importable() -> Path:
    root = Path(__file__).resolve().parents[1] / "third_party" / "dht"
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return root
```

- [ ] **Step 5: Run the test and verify GREEN**

Run: `python -m pytest tests/test_dht_vendor.py -v`

Expected: one passing test.

- [ ] **Step 6: Commit**

```bash
git add third_party/dht dht_simgcd/__init__.py tests/test_dht_vendor.py
git commit -m "build: vendor dHT source"
```

### Task 2: Implement foreground/background budget allocation

**Files:**
- Create: `dht_simgcd/budget.py`
- Test: `tests/test_dht_budget.py`

**Interfaces:**
- Produces: `TokenBudget(foreground: int, background: int)`.
- Produces: `allocate_token_budget(total_tokens, mode, foreground_fraction, foreground_ratio, foreground_density, min_foreground, min_background) -> TokenBudget`.

- [ ] **Step 1: Write failing tests for all allocation modes**

Tests must assert:

```python
assert allocate_token_budget(196, "none", .4, .75, 3., 16, 16) == TokenBudget(196, 0)
assert allocate_token_budget(196, "fixed", .4, .75, 3., 16, 16) == TokenBudget(147, 49)
assert allocate_token_budget(196, "area_weighted", .25, .75, 1., 16, 16) == TokenBudget(49, 147)
assert allocate_token_budget(196, "area_weighted", .25, .75, 3., 16, 16) == TokenBudget(98, 98)
assert allocate_token_budget(196, "area_weighted", 0., .75, 3., 16, 16) == TokenBudget(0, 196)
assert allocate_token_budget(196, "area_weighted", 1., .75, 3., 16, 16) == TokenBudget(196, 0)
```

Also test invalid modes, non-positive density, impossible minimum budgets, and
that every valid two-partition result sums to 196.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_dht_budget.py -v`

Expected: module import failure.

- [ ] **Step 3: Implement immutable budget allocation**

Use a frozen dataclass, validate inputs, calculate the specified formulas,
clip only when both partitions are non-empty, and preserve exact totals after
rounding.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_dht_budget.py -v`

Expected: all budget tests pass.

- [ ] **Step 5: Commit**

```bash
git add dht_simgcd/budget.py tests/test_dht_budget.py
git commit -m "feat: add mask-aware dHT token budgets"
```

### Task 3: Add mask path resolution, loading, and fallback accounting

**Files:**
- Create: `dht_simgcd/masks.py`
- Test: `tests/test_dht_masks.py`
- Reuse: `mask_utils.py`

**Interfaces:**
- Produces: `MaskLoadResult(mask: Tensor | None, missing: bool, invalid: bool)`.
- Produces: `TokenCutMaskStore(root, image_root=None, extensions=(".png", ".jpg", ".jpeg", ".npy"))`.
- Produces: `TokenCutMaskStore.load(image_path) -> MaskLoadResult`.
- Produces: counters `missing_count`, `invalid_count`, and `reset_counts()`.

- [ ] **Step 1: Write failing tests**

Use temporary directories to test matching relative paths, PNG and NPY masks,
binary conversion, missing files, corrupt files, empty masks, and counter
reset. Include an image path nested below a dataset root.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_dht_masks.py -v`

Expected: module import failure.

- [ ] **Step 3: Implement deterministic mask resolution and loading**

Resolve masks with the source image's relative path and alternate extensions.
Return a float tensor shaped `[1, H, W]`. Empty/corrupt/missing masks return
`None` and increment the appropriate counter.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_dht_masks.py -v`

Expected: all mask-store tests pass.

- [ ] **Step 5: Commit**

```bash
git add dht_simgcd/masks.py tests/test_dht_masks.py
git commit -m "feat: load TokenCut masks for dHT"
```

### Task 4: Build the fixed-budget mask-aware dHT tokenizer adapter

**Files:**
- Create: `dht_simgcd/tokenizer.py`
- Test: `tests/test_dht_tokenizer.py`

**Interfaces:**
- Produces: `RegionTokenBatch(tokens, valid_mask, segmentation, foreground_counts, background_counts)`.
- Produces: `MaskAwareDHTTokenizer(embed_dim=768, total_tokens=196, mask_mode="none", fg_ratio=.75, fg_density=3., min_fg_tokens=16, min_bg_tokens=16, trainable=True)`.
- Consumes: images `[B, 3, H, W]`, optional masks `[B, 1, H, W]`.
- Returns: token slots `[B, 196, 768]` and `valid_mask [B, 196]`.

- [ ] **Step 1: Write failing synthetic tests**

Use a small injected hierarchy provider rather than expensive real dHT to
assert:

- no-mask mode requests 196 regions;
- fixed and area-weighted modes request correct foreground/background counts;
- foreground and background candidates are selected independently;
- no cross-boundary region is created during budget reduction;
- output has 196 slots;
- padding is zero and marked invalid;
- actual foreground/background counts are reported;
- empty/full masks use one partition;
- gradients reach the region projection when trainable.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_dht_tokenizer.py -v`

Expected: adapter import failure.

- [ ] **Step 3: Implement the adapter around vendored dHT**

Separate:

```python
def compute_region_foreground_fraction(segmentation, masks): ...
def select_partition_regions(...): ...
def pad_region_tokens(tokens, total_tokens): ...
```

Inject the hierarchy provider for tests. The production provider uses
`dHTTokenizer`, `dHTExtractor`, and `dHTEmbedder`. Keep upstream code unchanged.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_dht_tokenizer.py -v`

Expected: all tokenizer tests pass.

- [ ] **Step 5: Run a real dHT CPU smoke test**

Run:

```bash
python -c "import torch; from dht_simgcd.tokenizer import MaskAwareDHTTokenizer; m=MaskAwareDHTTokenizer(embed_dim=32,total_tokens=16); y=m(torch.rand(1,3,32,32)); print(y.tokens.shape,y.valid_mask.sum().item())"
```

Expected: shape `[1, 16, 32]`; valid count between 1 and 16.

- [ ] **Step 6: Commit**

```bash
git add dht_simgcd/tokenizer.py tests/test_dht_tokenizer.py
git commit -m "feat: add fixed-budget mask-aware dHT tokenizer"
```

### Task 5: Adapt DINO v1 blocks to dHT region tokens and padding masks

**Files:**
- Create: `dht_simgcd/backbone.py`
- Test: `tests/test_dht_backbone.py`

**Interfaces:**
- Produces: `DHTDINOBackbone(dino_backbone, tokenizer)`.
- `forward(images, masks=None) -> Tensor[B, 768]`.
- Exposes `blocks`, `embed_dim`, and tokenizer statistics for training logs.

- [ ] **Step 1: Write failing wrapper tests**

Construct a tiny DINO-compatible fake with CLS token, two transformer blocks,
and norm. Assert output dimensions, padded-token invariance, CLS participation,
gradient flow, and `grad_from_block` trainability behavior.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_dht_backbone.py -v`

Expected: module import failure.

- [ ] **Step 3: Implement masked DINO execution**

Create CLS from the pretrained backbone. Add region geometry position features
from dHT embeddings. Adapt each DINO block through an explicit masked-attention
path that reuses pretrained `qkv`, projection, MLP, normalization, and residual
weights. Invalid padded keys and values cannot affect valid tokens; invalid
query outputs remain zero.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_dht_backbone.py -v`

Expected: all wrapper tests pass.

- [ ] **Step 5: Add an optional DINO v1 smoke test**

Mark it `@pytest.mark.network` and skip unless
`RUN_DINO_NETWORK_TESTS=1`. It loads `dino_vitb16`, processes a small batch,
and asserts `[B, 768]`.

- [ ] **Step 6: Commit**

```bash
git add dht_simgcd/backbone.py tests/test_dht_backbone.py
git commit -m "feat: run DINO v1 on dHT region tokens"
```

### Task 6: Integrate synchronized masks and dHT backbone into SimGCD training

**Files:**
- Create: `train_dht.py`
- Modify: `mask_utils.py`
- Test: `tests/test_train_dht.py`

**Interfaces:**
- CLI includes all mask and dHT options from the design.
- Produces: `build_dht_simgcd_model(args, dino_backbone=None)`.
- Produces: `set_dht_backbone_trainable(backbone, grad_from_block, tokenizer_trainable)`.
- Training batches support `(images, labels, indices, labelled_mask, masks)`.

- [ ] **Step 1: Write failing parser/model/batch tests**

Test defaults, all CLI modes, required `mask_root` when mode is not `none`,
DINO v1 repository/name, 768-dimensional head, trainability boundaries, and
missing-mask fallback. Use injected fake DINO and synthetic datasets.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_train_dht.py -v`

Expected: entry point import failure.

- [ ] **Step 3: Implement the dedicated entry point**

Reuse SimGCD training/evaluation functions rather than duplicating losses.
Extend existing paired transforms and masked dataset utilities so each
contrastive view receives its geometrically matching mask. Log token and mask
statistics once per epoch.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_train_dht.py -v`

Expected: all integration tests pass.

- [ ] **Step 5: Run a synthetic forward/backward smoke test**

Run the test fixture as a one-batch training step and confirm finite loss and
non-null tokenizer/projector gradients.

- [ ] **Step 6: Commit**

```bash
git add train_dht.py mask_utils.py tests/test_train_dht.py
git commit -m "feat: train SimGCD with dHT tokens"
```

### Task 7: Add environment, SLURM launch, and usage documentation

**Files:**
- Create: `requirements-dht.txt`
- Create: `scripts/train_dht_cub.slurm`
- Modify: `README.md`
- Test: `tests/test_dht_slurm.py`

**Interfaces:**
- SLURM script consumes `DATA_ROOT`, `MASK_ROOT`, `OUTPUT_ROOT`, and optional
  `PYTHON_BIN`.
- Script launches `train_dht.py` without installing dHT or accessing GitHub.

- [ ] **Step 1: Write failing static launch tests**

Assert that the script changes to `SLURM_SUBMIT_DIR`, exports repository-local
`PYTHONPATH`, uses `PYTHON_BIN`, passes `--mask_root`, selects DINO v1, and
contains no `pip install`, `git clone`, `wget`, or `curl`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_dht_slurm.py -v`

Expected: script missing.

- [ ] **Step 3: Add pinned environment and SLURM script**

Set minimum versions required by vendored dHT while retaining SimGCD runtime
packages. Use environment variables for cluster-specific paths.

- [ ] **Step 4: Document setup and experiments**

README must include environment creation, mask directory layout, no-mask,
fixed, and area-weighted commands, fallback behavior, and the fair baseline
matrix.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/test_dht_slurm.py -v`

Expected: all launch tests pass.

- [ ] **Step 6: Commit**

```bash
git add requirements-dht.txt scripts/train_dht_cub.slurm README.md tests/test_dht_slurm.py
git commit -m "docs: add SLURM workflow for dHT SimGCD"
```

### Task 8: Full verification and branch handoff

**Files:**
- Modify only files required to fix verification failures.

**Interfaces:**
- Produces a branch that is reproducible, tested, and ready for remote use.

- [ ] **Step 1: Run the complete offline test suite**

Run: `python -m pytest tests -v -m "not network"`

Expected: all tests pass, network test skipped.

- [ ] **Step 2: Run compile and formatting checks**

Run:

```bash
python -m compileall dht_simgcd train_dht.py third_party/dht/dht
git diff --check origin/main...HEAD
```

Expected: exit code zero.

- [ ] **Step 3: Run CPU smoke tests**

Exercise `none`, `fixed`, and `area_weighted` modes with synthetic inputs.
Each output must contain 196 visual slots and finite forward/backward values.

- [ ] **Step 4: Inspect branch state**

Run:

```bash
git status --short --branch
git log --oneline --decorate origin/main..HEAD
```

Expected: clean worktree and the planned commits.

- [ ] **Step 5: Push the completed branch**

```bash
git push origin feat/dht-simgcd-baseline
```

