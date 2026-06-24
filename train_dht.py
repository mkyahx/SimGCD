"""
Train SimGCD with dHT region tokens and a DINO v1 ViT-B/16 backbone.

The visual-token budget is fixed at 196. When TokenCut masks are enabled,
foreground and background are tokenized independently so foreground can retain
more regions while background is represented more coarsely.

Example:
    python train_dht.py \
        --dataset_name cub \
        --dht_root /path/to/dHT \
        --dht_mask_mode area_weighted \
        --dht_fg_density 3.0 \
        --mask_root /path/to/tokencut_masks
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import exp_root
from data.augmentations import get_transform
from data.get_datasets import get_class_splits, get_datasets
from last_train_utils import train_with_optional_masks
from mask_utils import (
    MaskedContrastiveLearningViewGenerator,
    MaskedDataset,
    MaskedModelWithHead,
    PairedImagenetTransform,
    clear_image_transform,
)
from model import ContrastiveLearningViewGenerator, DINOHead
from train_last_1d_channel import (
    ensure_runtime_tmpdir,
    load_backbone,
    make_loader_kwargs,
    seed_worker,
    set_backbone_trainable,
    set_random_seed,
)
from util.general_utils import init_experiment


@dataclass(frozen=True)
class TokenBudget:
    foreground: int
    background: int

    @property
    def total(self):
        return self.foreground + self.background


def allocate_token_budget(
    total_tokens,
    mode,
    foreground_fraction,
    foreground_ratio,
    foreground_density,
    min_foreground,
    min_background,
):
    """Allocate an exact visual-token budget between foreground/background."""
    total_tokens = int(total_tokens)
    foreground_fraction = float(foreground_fraction)

    if total_tokens <= 0:
        raise ValueError("--dht_num_tokens must be positive.")
    if mode not in {"none", "fixed", "area_weighted"}:
        raise ValueError(f"Unknown dHT mask mode: {mode}")
    if not 0.0 <= foreground_fraction <= 1.0:
        raise ValueError("Foreground fraction must be in [0, 1].")

    if mode == "none" or foreground_fraction >= 1.0:
        return TokenBudget(total_tokens, 0)
    if foreground_fraction <= 0.0:
        return TokenBudget(0, total_tokens)

    min_foreground = int(min_foreground)
    min_background = int(min_background)
    if min_foreground < 0 or min_background < 0:
        raise ValueError("Minimum token budgets cannot be negative.")
    if min_foreground + min_background > total_tokens:
        raise ValueError("Minimum foreground/background budgets exceed total.")

    if mode == "fixed":
        if not 0.0 <= foreground_ratio <= 1.0:
            raise ValueError("--dht_fg_ratio must be in [0, 1].")
        foreground = round(total_tokens * float(foreground_ratio))
    else:
        foreground_density = float(foreground_density)
        if foreground_density <= 0:
            raise ValueError("--dht_fg_density must be positive.")
        weighted_foreground = foreground_density * foreground_fraction
        denominator = weighted_foreground + 1.0 - foreground_fraction
        foreground = round(total_tokens * weighted_foreground / denominator)

    foreground = max(min_foreground, foreground)
    foreground = min(total_tokens - min_background, foreground)
    return TokenBudget(foreground, total_tokens - foreground)


def configure_dht_import(dht_root):
    """Add a local dHT checkout/package root without requiring network access."""
    candidates = []
    if dht_root:
        candidates.append(os.path.abspath(os.path.expanduser(dht_root)))
    vendored = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "dht")
    candidates.append(vendored)

    for candidate in candidates:
        package_dir = os.path.join(candidate, "dht")
        if os.path.isfile(os.path.join(package_dir, "__init__.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate

    raise FileNotFoundError(
        "Cannot find the dHT Python package. Pass --dht_root /path/to/dHT "
        "or vendor it at third_party/dht."
    )


class DHTRegionTokenizer(nn.Module):
    """Convert images into a fixed number of dHT region-token slots."""

    def __init__(
        self,
        embed_dim=768,
        total_tokens=196,
        patch_size=16,
        mask_mode="none",
        foreground_ratio=0.75,
        foreground_density=3.0,
        min_foreground=16,
        min_background=16,
        tokenizer_hidden=3,
        trainable=True,
    ):
        super().__init__()
        from dht.tok.embedder import dHTEmbedder
        from dht.tok.extractor import dHTExtractor
        from dht.tok.tokenizer import dHTTokenizer

        self.embed_dim = int(embed_dim)
        self.total_tokens = int(total_tokens)
        self.mask_mode = mask_mode
        self.foreground_ratio = float(foreground_ratio)
        self.foreground_density = float(foreground_density)
        self.min_foreground = int(min_foreground)
        self.min_background = int(min_background)

        self.tokenizer = dHTTokenizer(
            in_ch=3,
            hid_ch=int(tokenizer_hidden),
            compute_grad=False,
        )
        self.extractor = dHTExtractor(
            patch_size=int(patch_size),
            channels=3,
        )
        self.embedder = dHTEmbedder(
            embed_dim=self.embed_dim,
            patch_size=int(patch_size),
            channels=3,
            compute_grad=False,
            num_cls_tokens=1,
        )

        if not trainable:
            self.requires_grad_(False)

    @staticmethod
    def _normalize_masks(masks, images):
        if masks is None:
            return None
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(
                f"Expected masks shaped [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}"
            )
        if masks.shape[-2:] != images.shape[-2:]:
            masks = nn.functional.interpolate(
                masks.float(),
                size=images.shape[-2:],
                mode="nearest",
            )
        return masks.to(device=images.device, dtype=images.dtype).gt(0.5)

    @staticmethod
    def _fill_outside(image, keep_mask):
        keep_mask = keep_mask.to(dtype=image.dtype)
        denominator = keep_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mean = (image * keep_mask).sum(dim=(-2, -1), keepdim=True) / denominator
        image_mean = image.mean(dim=(-2, -1), keepdim=True)
        nonempty = keep_mask.sum(dim=(-2, -1), keepdim=True).gt(0)
        mean = torch.where(nonempty, mean, image_mean)
        return image * keep_mask + mean * (1.0 - keep_mask)

    def _tokenize(self, images, target):
        if target <= 0:
            return images.new_zeros(images.shape[0], 0, self.embed_dim)

        tokenizer_result = self.tokenizer(
            images,
            final_merging=True,
            target=int(target),
        )
        embedded = self.embedder(self.extractor(tokenizer_result))
        region_tokens = embedded.fV[:, 1:]
        valid = embedded.amask[:, 1:]

        outputs = []
        for sample_tokens, sample_valid in zip(region_tokens, valid):
            sample_tokens = sample_tokens[sample_valid]
            if sample_tokens.shape[0] >= target:
                sample_tokens = sample_tokens[:target]
            else:
                padding = sample_tokens.new_zeros(
                    target - sample_tokens.shape[0],
                    self.embed_dim,
                )
                sample_tokens = torch.cat([sample_tokens, padding], dim=0)
            outputs.append(sample_tokens)
        return torch.stack(outputs)

    def _tokenize_masked_sample(self, image, mask):
        foreground_fraction = mask.float().mean().item()
        budget = allocate_token_budget(
            total_tokens=self.total_tokens,
            mode=self.mask_mode,
            foreground_fraction=foreground_fraction,
            foreground_ratio=self.foreground_ratio,
            foreground_density=self.foreground_density,
            min_foreground=self.min_foreground,
            min_background=self.min_background,
        )

        branches = []
        if budget.foreground:
            foreground_image = self._fill_outside(image, mask)
            branches.append(self._tokenize(foreground_image, budget.foreground))
        if budget.background:
            background_image = self._fill_outside(image, ~mask)
            branches.append(self._tokenize(background_image, budget.background))

        tokens = torch.cat(branches, dim=1)
        if tokens.shape[1] != self.total_tokens:
            raise RuntimeError(
                f"dHT produced {tokens.shape[1]} slots; expected {self.total_tokens}."
            )
        return tokens

    def forward(self, images, masks=None):
        masks = self._normalize_masks(masks, images)
        if self.mask_mode == "none" or masks is None:
            return self._tokenize(images, self.total_tokens)

        outputs = []
        # Area-weighted budgets can differ per image, so process samples
        # independently. Fixed-ratio mode uses the same path for consistency.
        for image, mask in zip(images, masks):
            outputs.append(
                self._tokenize_masked_sample(
                    image.unsqueeze(0),
                    mask.unsqueeze(0),
                )
            )
        return torch.cat(outputs, dim=0)


class DHTDINOBackbone(nn.Module):
    """Reuse pretrained DINO v1 transformer blocks over dHT region tokens."""

    def __init__(self, backbone, tokenizer, mode="fusion", alpha=0.1):
        super().__init__()
        if mode not in {"cls", "replace", "fusion"}:
            raise ValueError(f"Unknown dHT mode: {mode}")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("--dht_alpha must be in [0, 1].")
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.mode = mode
        self.alpha = float(alpha)
        self.embed_dim = int(
            getattr(backbone, "embed_dim", getattr(backbone, "num_features", 768))
        )

    def forward_replace(self, images, masks=None):
        region_tokens = self.tokenizer(images, masks)
        cls = self.backbone.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat([cls, region_tokens], dim=1)

        # dHT embeddings already contain region geometry. Do not add the DINO
        # fixed-grid positional embedding, which assumes a 14x14 patch grid.
        if hasattr(self.backbone, "pos_drop"):
            tokens = self.backbone.pos_drop(tokens)
        for block in self.backbone.blocks:
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)
        return tokens[:, 0]

    def forward(self, images, masks=None):
        cls_feature = self.backbone(images)
        if self.mode == "cls":
            return cls_feature

        dht_feature = self.forward_replace(images, masks)
        if self.mode == "replace":
            return dht_feature

        return (1.0 - self.alpha) * cls_feature + self.alpha * dht_feature


def build_parser():
    parser = argparse.ArgumentParser(
        description="SimGCD with dHT region tokens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--eval_funcs", nargs="+", default=["v2", "v2p"])
    parser.add_argument("--warmup_model_dir", default=None, type=str)
    parser.add_argument("--dataset_name", default="scars", type=str)
    parser.add_argument("--prop_train_labels", default=0.5, type=float)
    parser.add_argument("--use_ssb_splits", action="store_true", default=True)
    parser.add_argument("--grad_from_block", default=11, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--gamma", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--exp_root", default=exp_root, type=str)
    parser.add_argument("--transform", default="imagenet", type=str)
    parser.add_argument("--sup_weight", default=0.35, type=float)
    parser.add_argument("--n_views", default=2, type=int)
    parser.add_argument("--memax_weight", default=2.0, type=float)
    parser.add_argument("--warmup_teacher_temp", default=0.07, type=float)
    parser.add_argument("--teacher_temp", default=0.04, type=float)
    parser.add_argument("--warmup_teacher_temp_epochs", default=30, type=int)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--tmp_dir", default=None, type=str)
    parser.add_argument("--pin_memory", default=0, type=int)
    parser.add_argument("--persistent_workers", default=0, type=int)
    parser.add_argument("--mp_sharing_strategy", default="file_system", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no_seed", action="store_const", const=None, dest="seed")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")

    parser.add_argument("--backbone_repo", default="facebookresearch/dino:main")
    parser.add_argument("--backbone_name", default="dino_vitb16")
    parser.add_argument("--backbone_source", default=None, choices=[None, "github", "local"])
    parser.add_argument("--backbone_weights", default=None)
    parser.add_argument("--image_size", default=224, type=int)

    parser.add_argument("--dht_root", default=os.environ.get("DHT_ROOT"))
    parser.add_argument("--dht_num_tokens", default=196, type=int)
    parser.add_argument("--dht_patch_size", default=16, type=int)
    parser.add_argument(
        "--dht_mask_mode",
        default="none",
        choices=["none", "fixed", "area_weighted"],
    )
    parser.add_argument("--dht_fg_ratio", default=0.75, type=float)
    parser.add_argument("--dht_fg_density", default=3.0, type=float)
    parser.add_argument("--dht_min_fg_tokens", default=16, type=int)
    parser.add_argument("--dht_min_bg_tokens", default=16, type=int)
    parser.add_argument("--dht_tokenizer_hidden", default=3, type=int)
    parser.add_argument("--freeze_dht", action="store_true", default=False)
    parser.add_argument(
        "--dht_mode",
        default="fusion",
        choices=["cls", "replace", "fusion"],
        help="How dHT is applied: cls keeps original DINO CLS, replace uses dHT tokens, fusion blends both.",
    )
    parser.add_argument("--dht_alpha", default=0.1, type=float)

    parser.add_argument("--mask_root", default=None, type=str)
    parser.add_argument("--mask_image_root", default=None, type=str)
    parser.add_argument("--mask_threshold", default=0.5, type=float)
    parser.add_argument(
        "--missing_mask",
        default="zeros",
        choices=["error", "zeros", "ones"],
    )
    return parser


def validate_args(args):
    if args.dht_num_tokens != 196:
        raise ValueError("This baseline fixes --dht_num_tokens at 196.")
    if not 0.0 <= args.dht_alpha <= 1.0:
        raise ValueError("--dht_alpha must be in [0, 1].")
    if args.dht_mask_mode != "none" and args.mask_root is None:
        raise ValueError("--mask_root is required when dHT mask mode is enabled.")
    allocate_token_budget(
        total_tokens=args.dht_num_tokens,
        mode=args.dht_mask_mode,
        foreground_fraction=0.5,
        foreground_ratio=args.dht_fg_ratio,
        foreground_density=args.dht_fg_density,
        min_foreground=args.dht_min_fg_tokens,
        min_background=args.dht_min_bg_tokens,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    configure_dht_import(args.dht_root)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda:0")
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    args.interpolation = 3
    args.crop_pct = 0.875

    init_experiment(args, runner_name=["simgcd_dht"])
    ensure_runtime_tmpdir(args)
    torch.backends.cudnn.benchmark = args.seed is None or not args.deterministic

    backbone = load_backbone(args)
    set_backbone_trainable(backbone, args.grad_from_block)
    if int(getattr(backbone, "embed_dim", 768)) != 768:
        raise ValueError("Expected DINO v1 ViT-B/16 with embed_dim=768.")

    train_transform, test_transform = get_transform(
        args.transform,
        image_size=args.image_size,
        args=args,
    )
    use_masks = args.dht_mask_mode != "none"
    if use_masks:
        paired_train_transform = MaskedContrastiveLearningViewGenerator(
            PairedImagenetTransform(
                image_size=args.image_size,
                interpolation=args.interpolation,
                crop_pct=args.crop_pct,
                train=True,
            ),
            n_views=args.n_views,
        )
        paired_test_transform = PairedImagenetTransform(
            image_size=args.image_size,
            interpolation=args.interpolation,
            crop_pct=args.crop_pct,
            train=False,
        )
        dataset_train_transform = None
        dataset_test_transform = None
    else:
        dataset_train_transform = ContrastiveLearningViewGenerator(
            base_transform=train_transform,
            n_views=args.n_views,
        )
        dataset_test_transform = test_transform

    train_dataset, _, unlabelled_test, _ = get_datasets(
        args.dataset_name,
        dataset_train_transform,
        dataset_test_transform,
        args,
    )
    if use_masks:
        clear_image_transform(train_dataset)
        clear_image_transform(unlabelled_test)
        train_dataset = MaskedDataset(
            train_dataset,
            mask_root=args.mask_root,
            image_root=args.mask_image_root,
            mask_size=args.image_size,
            threshold=args.mask_threshold,
            missing=args.missing_mask,
            transform=paired_train_transform,
        )
        unlabelled_test = MaskedDataset(
            unlabelled_test,
            mask_root=args.mask_root,
            image_root=args.mask_image_root,
            mask_size=args.image_size,
            threshold=args.mask_threshold,
            missing=args.missing_mask,
            transform=paired_test_transform,
        )

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [
        1.0 if index < label_len else label_len / unlabelled_len
        for index in range(len(train_dataset))
    ]
    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=generator,
        **make_loader_kwargs(args),
    )
    test_loader = DataLoader(
        unlabelled_test,
        batch_size=256,
        shuffle=False,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=generator,
        **make_loader_kwargs(args, pin_memory=False, persistent_workers=False),
    )

    tokenizer = DHTRegionTokenizer(
        embed_dim=768,
        total_tokens=args.dht_num_tokens,
        patch_size=args.dht_patch_size,
        mask_mode=args.dht_mask_mode,
        foreground_ratio=args.dht_fg_ratio,
        foreground_density=args.dht_fg_density,
        min_foreground=args.dht_min_fg_tokens,
        min_background=args.dht_min_bg_tokens,
        tokenizer_hidden=args.dht_tokenizer_hidden,
        trainable=not args.freeze_dht,
    )
    dht_backbone = DHTDINOBackbone(
        backbone,
        tokenizer,
        mode=args.dht_mode,
        alpha=args.dht_alpha,
    )
    projector = DINOHead(
        in_dim=768,
        out_dim=args.mlp_out_dim,
        nlayers=args.num_mlp_layers,
    )
    model = MaskedModelWithHead(dht_backbone, projector).to(device)

    args.logger.info(
        "dHT-SimGCD: DINO=%s tokens=%d mode=%s alpha=%.3f mask_mode=%s fg_ratio=%.3f fg_density=%.3f"
        % (
            args.backbone_name,
            args.dht_num_tokens,
            args.dht_mode,
            args.dht_alpha,
            args.dht_mask_mode,
            args.dht_fg_ratio,
            args.dht_fg_density,
        )
    )
    train_with_optional_masks(model, train_loader, None, test_loader, args)


if __name__ == "__main__":
    main()
