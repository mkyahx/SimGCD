# dHT–SimGCD Baseline Design

## Goal

Integrate Differentiable Hierarchical Visual Tokenization (dHT) into the
`mkyahx/SimGCD` repository while retaining the DINO v1 ViT-B/16 transformer,
SimGCD objectives, and evaluation protocol. The baseline must use exactly 196
visual region tokens per image so that its transformer compute budget is
comparable to the original `16×16` patch tokenizer.

An optional precomputed TokenCut mask redistributes the fixed token budget:
foreground regions remain finer and background regions become coarser. The
background is compressed rather than removed.

## Scope

The first implementation provides:

- vendored dHT source code and its MIT license;
- a DINO v1 ViT-B/16 wrapper that consumes padded dHT region tokens;
- exactly 196 valid region tokens per image, plus one CLS token;
- ordinary dHT tokenization when masks are disabled;
- fixed-ratio and area-weighted foreground/background token budgets;
- precomputed mask loading through `--mask_root` during training and testing;
- a separate `train_dht.py` entry point;
- a SLURM launch script;
- focused unit and integration tests.

The first implementation does not:

- run TokenCut online;
- alter SimGCD losses, prototypes, evaluation, or data splits;
- train a new transformer backbone from scratch;
- support adaptive total token counts;
- add MGCE;
- remove background tokens.

## Repository Layout

```text
SimGCD/
├── third_party/
│   └── dht/
│       ├── dht/
│       ├── LICENSE
│       └── SOURCE.md
├── dht_simgcd/
│   ├── __init__.py
│   ├── budget.py
│   ├── tokenizer.py
│   ├── backbone.py
│   └── masks.py
├── tests/
│   ├── test_dht_budget.py
│   ├── test_dht_tokenizer.py
│   ├── test_dht_backbone.py
│   └── test_dht_masks.py
├── train_dht.py
└── scripts/
    └── train_dht_cub.slurm
```

`third_party/dht` preserves the upstream package structure. SimGCD-specific
behavior lives in `dht_simgcd` so upstream code remains easy to compare and
update.

## Runtime and Dependency Strategy

dHT currently requires Python 3.10+, PyTorch 2.7+, torchvision 0.22+, NumPy
2.2+, and Pillow 12+. This branch therefore uses a separate environment from
the original SimGCD environment.

The vendored package avoids network access, Git submodule initialization, and
runtime `pip install git+...` calls on SLURM compute nodes. The launch script
adds the repository root and `third_party/dht` to `PYTHONPATH`.

## Model Architecture

```text
image
  → dHT hierarchy and region features
  → optional mask-aware foreground/background budget allocation
  → exactly 196 region embeddings
  → DINO v1 CLS token + DINO positional encoding adapted to region geometry
  → pretrained DINO v1 ViT-B/16 transformer blocks
  → normalized CLS representation
  → unchanged SimGCD DINOHead and training objectives
```

The original DINO patch projection is replaced, but its CLS token, transformer
blocks, normalization layer, and pretrained weights are retained.

Each dHT region token uses the vendored extractor/embedder path and is
projected to 768 dimensions. Positional information is derived from region
geometry rather than a fixed `14×14` grid. The wrapper exposes a normal
`forward(images) -> Tensor[B, 768]` interface so it can be placed before the
existing `DINOHead`.

## Fixed Token Budget

The total visual-token budget is:

```text
N = 196
```

The CLS token is separate and does not count toward `N`.

Without masks, dHT performs its normal hierarchical construction and is pruned
or merged to exactly 196 regions.

With masks, foreground and background are assigned separate budgets and are
merged independently until both budgets are met. Cross-boundary merging is
disabled during this budget-reduction stage. The resulting foreground and
background region tokens are concatenated into one sequence and processed by
global DINO attention.

If an image cannot supply the requested number of distinct regions in one
partition, the unused budget is transferred to the other partition. If both
partitions together contain fewer than 196 regions, deterministic zero-padding
is applied and excluded through the attention mask. This is an exceptional
fallback; ordinary `224×224` inputs should provide sufficient initial regions.

## Mask Budget Modes

CLI:

```text
--dht_mask_mode none|fixed|area_weighted
--dht_num_tokens 196
--dht_fg_ratio 0.75
--dht_fg_density 3.0
--dht_min_fg_tokens 16
--dht_min_bg_tokens 16
--mask_root /path/to/tokencut_masks
```

### No mask

`none` ignores `mask_root` and assigns all 196 tokens through ordinary dHT
selection.

### Fixed ratio

```text
N_fg = round(N × dht_fg_ratio)
N_bg = N - N_fg
```

The minimum foreground/background budgets are enforced before final rounding.

### Area weighted

Let `A` be the foreground-pixel fraction and `α = dht_fg_density`:

```text
N_fg = round(N × αA / (αA + 1 - A))
N_bg = N - N_fg
```

`α=1` allocates tokens by area. `α>1` gives foreground pixels greater token
density. Minimum foreground/background budgets are enforced when both
partitions are non-empty.

Empty and full masks degrade to a single-partition 196-token allocation.

## Mask Loading and Augmentation

Masks are precomputed and stored below `--mask_root` using the same relative
path layout as the image dataset. The existing repository mask utilities are
reused where possible.

Training applies identical crop, resize, and flip parameters to each image and
its mask. Test-time masks follow the deterministic test transform.

Missing, unreadable, or empty masks cause that sample to use ordinary
mask-free dHT. Missing-mask counts are accumulated and logged at the end of
each epoch; fallback must never be silent.

Masks are converted to binary values after geometric transformation. Their
values do not enter the RGB image or DINO feature channels.

## DINO v1 Compatibility

The baseline uses:

```text
facebookresearch/dino:main
dino_vitb16
feature dimension = 768
```

All DINO parameters are initially frozen. Parameters in blocks at or after
`--grad_from_block` are trainable, matching SimGCD behavior. The dHT tokenizer
and region projection have their own trainability flag:

```text
--dht_trainable
```

The default baseline keeps dHT trainable because downstream gradients are part
of the method's definition. A frozen-tokenizer ablation is supported.

The DINO wrapper must pass a key-padding/attention mask to every transformer
block so padded region tokens cannot affect valid tokens or CLS. For the
standard 196-token path no padding is expected.

## Training and Evaluation

`train_dht.py` reuses the existing:

- dataset splits;
- balanced labeled/unlabeled sampler;
- two-view contrastive augmentation;
- supervised and unsupervised contrastive losses;
- cross-view self-distillation;
- mean-entropy regularization;
- prototype head;
- All/Old/New evaluation.

Only backbone construction and optional mask-bearing batches differ from the
standard SimGCD path.

The first comparison matrix is:

| Run | Tokenizer | Mask mode | Tokens |
| --- | --- | --- | ---: |
| SimGCD | Patch-16 | none | 196 |
| dHT-SimGCD | dHT | none | 196 |
| dHT-SimGCD | dHT | fixed | 196 |
| dHT-SimGCD | dHT | area-weighted | 196 |

Report All/Old/New accuracy, mean foreground/background region counts,
throughput, peak memory, and missing-mask count.

## SLURM Interface

The launch script:

- changes to `SLURM_SUBMIT_DIR`;
- sets repository-local `PYTHONPATH`;
- accepts dataset, data root, mask root, output root, and environment path as
  environment variables;
- never downloads code or model dependencies on a compute node;
- writes logs and checkpoints under the requested output root.

Example:

```bash
MASK_ROOT=/datasets/cub/tokencut_masks \
DATA_ROOT=/datasets \
sbatch scripts/train_dht_cub.slurm
```

## Testing Strategy

Unit tests cover:

- fixed-ratio budget allocation;
- area-weighted allocation at `α=1` and `α>1`;
- minimum-budget clipping;
- empty/full masks;
- exact total budget of 196;
- missing-mask fallback and accounting;
- synchronized image/mask transforms;
- foreground and background region-budget enforcement;
- deterministic padding and attention masks;
- 768-dimensional DINO wrapper output.

An integration smoke test uses synthetic images and a tiny/stub transformer
with the same interface. A full DINO smoke test is optional and marked as
network/GPU dependent.

## Risks and Mitigations

- **Upstream dHT is alpha-stage:** isolate modifications outside vendored
  files and record the exact source commit.
- **Dependency jump:** provide a dedicated environment file rather than
  mutating the original historical environment silently.
- **DINO positional mismatch:** encode region geometry explicitly and test
  permutation, shape, and padding behavior.
- **Mask noise:** masks only redistribute capacity; they do not remove
  background or supervise class labels.
- **Mask boundary errors:** foreground/background separation is used during
  budget reduction only; global attention can still exchange information.
- **Unequal compute:** enforce exactly 196 visual tokens in every reported
  baseline.

## Acceptance Criteria

The implementation is complete when:

1. all unit tests pass;
2. a CPU synthetic smoke test executes forward and backward;
3. a DINO v1 forward smoke test returns `[B, 768]`;
4. mask-free, fixed, and area-weighted modes each produce exactly 196 valid
   visual tokens;
5. missing masks produce logged fallback rather than failure;
6. the SLURM script imports vendored dHT without network access;
7. repository documentation includes environment creation and launch commands;
8. `git diff --check` passes.
