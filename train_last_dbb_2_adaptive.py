import argparse
import os
import random
import tempfile
import time

import numpy as np
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
from model_last_dbb_2_adaptive import LASTDBB2AdaptiveBackbone
from util.general_utils import init_experiment


def set_random_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_runtime_tmpdir(args):
    tmp_root = str(getattr(args, "tmp_dir", "") or "").strip()
    if not tmp_root:
        tmp_root = os.environ.get("SLURM_TMPDIR", "").strip()
    if not tmp_root:
        tmp_root = "/tmp"

    tmp_root = os.path.abspath(os.path.expanduser(tmp_root))
    os.makedirs(tmp_root, exist_ok=True)

    unique_name = f"simgcd_last_dbb_2_adaptive_{os.getpid()}_{int(time.time())}"
    tmpdir = os.path.join(tmp_root, unique_name)
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    os.environ["TMP"] = tmpdir
    os.environ["TEMP"] = tmpdir
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(tmpdir, "torch_extensions")
    os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(tmpdir, "pycache")
    tempfile.tempdir = tmpdir
    args.tmp_dir = tmpdir
    args.logger.info(f"Using TMPDIR={tmpdir}")

    try:
        torch.multiprocessing.set_sharing_strategy(str(getattr(args, "mp_sharing_strategy", "file_system")))
        args.logger.info(f"Using torch multiprocessing sharing strategy: {torch.multiprocessing.get_sharing_strategy()}")
    except Exception as exc:
        args.logger.warning(f"Failed to set multiprocessing sharing strategy: {exc}")


def make_loader_kwargs(args, pin_memory: bool = None, persistent_workers: bool = None):
    num_workers = int(args.num_workers)
    pin_memory = bool(args.pin_memory if pin_memory is None else pin_memory)
    persistent_workers = bool(args.persistent_workers if persistent_workers is None else persistent_workers)
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and (num_workers > 0),
    }


def load_backbone(args):
    hub_kwargs = {}
    source = None
    if args.backbone_source:
        source = args.backbone_source
    if args.backbone_weights:
        hub_kwargs["weights"] = args.backbone_weights

    if source is None:
        backbone = torch.hub.load(args.backbone_repo, args.backbone_name, **hub_kwargs)
    else:
        backbone = torch.hub.load(args.backbone_repo, args.backbone_name, source=source, **hub_kwargs)

    if args.warmup_model_dir is not None:
        args.logger.info(f"Loading weights from {args.warmup_model_dir}")
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location="cpu"))
    return backbone


def parse_block_number(name):
    parts = name.split(".")
    for marker in ("blocks", "block"):
        if marker in parts:
            idx = parts.index(marker) + 1
            if idx < len(parts) and parts[idx].isdigit():
                return int(parts[idx])
    return None


def set_backbone_trainable(backbone, grad_from_block):
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    for name, parameter in backbone.named_parameters():
        block_num = parse_block_number(name)
        if block_num is not None and block_num >= grad_from_block:
            parameter.requires_grad = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--eval_funcs", nargs="+", help="Which eval functions to use", default=["v2", "v2p"])

    parser.add_argument("--warmup_model_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="scars")
    parser.add_argument("--prop_train_labels", type=float, default=0.5)
    parser.add_argument("--use_ssb_splits", action="store_true", default=True)

    parser.add_argument("--grad_from_block", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--exp_root", type=str, default=exp_root)
    parser.add_argument("--transform", type=str, default="imagenet")
    parser.add_argument("--sup_weight", type=float, default=0.35)
    parser.add_argument("--n_views", default=2, type=int)

    parser.add_argument("--memax_weight", type=float, default=2)
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
    parser.add_argument("--seed", type=int, default=0, help="Set to enable reproducible training.")
    parser.add_argument("--no_seed", action="store_const", const=None, dest="seed",
                        help="Disable explicit random seeding.")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic CUDA/PyTorch behavior when --seed is set.")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic",
                        help="Allow non-deterministic CUDA/PyTorch behavior.")

    parser.add_argument("--backbone_repo", default="facebookresearch/dino:main", type=str)
    parser.add_argument("--backbone_name", default="dino_vitb16", type=str)
    parser.add_argument("--backbone_source", default=None, type=str, choices=[None, "github", "local"])
    parser.add_argument("--backbone_weights", default=None, type=str)
    parser.add_argument("--image_size", default=224, type=int)

    parser.add_argument("--sf_topk", default=4, type=int)
    parser.add_argument("--sf_vote_topk", default=None, type=int)
    parser.add_argument("--sf_eps", default=1e-6, type=float)
    parser.add_argument("--sf_mask_mode", default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--sf_sigma", default=None, type=float)
    parser.add_argument("--sf_adaptive_k", default=1.0, type=float)
    parser.add_argument("--last_token_source", default="patch", choices=["patch", "all"])
    parser.add_argument("--use_mask", action="store_true", default=False,
                        help="Restrict LAST token selection to foreground patches from .npy masks.")
    parser.add_argument("--mask_root", default=None, type=str,
                        help="Root directory for .npy masks with the same relative layout as images.")
    parser.add_argument("--mask_image_root", default=None, type=str,
                        help="Optional image root whose internal relative layout matches --mask_root. "
                             "If omitted or incompatible, the inferred dataset image root is used.")
    parser.add_argument("--mask_threshold", default=0.5, type=float,
                        help="Threshold used to binarize loaded masks.")
    parser.add_argument("--missing_mask", default="error", choices=["error", "zeros", "ones"],
                        help="Behavior when a mask file is missing.")

    args = parser.parse_args()
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda:0")
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=["simgcd_last_dbb_2_adaptive"])
    ensure_runtime_tmpdir(args)
    args.logger.info(f"Using evaluation function {args.eval_funcs[0]} to print results")

    torch.backends.cudnn.benchmark = args.seed is None or not args.deterministic

    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = load_backbone(args)
    args.feat_dim = int(getattr(backbone, "embed_dim", getattr(backbone, "num_features", 768)))
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    set_backbone_trainable(backbone, args.grad_from_block)

    args.logger.info(
        "Building LAST DBB 2D Adaptive backbone=%s repo=%s feat_dim=%s topk=%s vote_topk=%s mask_mode=%s sigma=%s adaptive_k=%s"
        % (
            args.backbone_name,
            args.backbone_repo,
            args.feat_dim,
            args.sf_topk,
            args.sf_vote_topk if args.sf_vote_topk is not None else args.sf_topk,
            args.sf_mask_mode,
            args.sf_sigma if args.sf_sigma is not None else args.feat_dim ** 0.5,
            args.sf_adaptive_k,
        )
    )

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name,
        train_transform,
        test_transform,
        args,
    )
    if args.use_mask:
        if args.mask_root is None:
            raise ValueError("--mask_root is required when --use_mask is set.")
        clear_image_transform(train_dataset)
        clear_image_transform(unlabelled_train_examples_test)
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
        train_dataset = MaskedDataset(
            train_dataset,
            mask_root=args.mask_root,
            image_root=args.mask_image_root,
            mask_size=args.image_size,
            threshold=args.mask_threshold,
            missing=args.missing_mask,
            transform=paired_train_transform,
        )
        unlabelled_train_examples_test = MaskedDataset(
            unlabelled_train_examples_test,
            mask_root=args.mask_root,
            image_root=args.mask_image_root,
            mask_size=args.image_size,
            threshold=args.mask_threshold,
            missing=args.missing_mask,
            transform=paired_test_transform,
        )
        args.logger.info(f"Using foreground masks from {args.mask_root}")

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    loader_generator = None
    if args.seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_dataset),
        generator=loader_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=loader_generator,
        **make_loader_kwargs(args),
    )
    test_loader_unlabelled = DataLoader(
        unlabelled_train_examples_test,
        batch_size=256,
        shuffle=False,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=loader_generator,
        **make_loader_kwargs(args, pin_memory=False, persistent_workers=False),
    )

    sf_backbone = LASTDBB2AdaptiveBackbone(
        backbone=backbone,
        topk=args.sf_topk,
        vote_topk=args.sf_vote_topk,
        eps=args.sf_eps,
        token_source=args.last_token_source,
        mask_mode=args.sf_mask_mode,
        sigma=args.sf_sigma,
        adaptive_k=args.sf_adaptive_k,
    )
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = MaskedModelWithHead(sf_backbone, projector).to(device)

    train_with_optional_masks(model, train_loader, None, test_loader_unlabelled, args)
