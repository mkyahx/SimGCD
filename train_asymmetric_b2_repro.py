import argparse
import math
import os

_cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _cublas_workspace_config not in (None, ":4096:8"):
    raise RuntimeError(
        "Strict reproducibility requires CUBLAS_WORKSPACE_CONFIG=:4096:8, "
        f"got {_cublas_workspace_config!r}."
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader

from asymmetric_b2_mask_model import B2ForegroundPairMaskModel
from config import exp_root
from data.get_datasets import get_class_splits
from data.paired_mask_transforms import PairedMaskViewGenerator, get_paired_mask_transform
from model import DINOHead, DistillLoss, SupConLoss, get_params_groups, info_nce_logits
from train_asymmetric_mask_repro import (
    AverageMeter,
    build_mask_datasets,
    collate_eval_mask_batch,
    collate_train_mask_batch,
    init_experiment,
    partial,
    seed_worker,
    set_strict_seed,
    test,
)


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )
    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab, patch_mask = batch
            mask_lab = mask_lab[:, 0]
            class_labels = class_labels.cuda(non_blocking=True)
            mask_lab = mask_lab.cuda(non_blocking=True).bool()
            if len(images) != 2 or len(patch_mask) != 2:
                raise ValueError("B2 mask training requires exactly two image/mask views.")
            images = [image.cuda(non_blocking=True) for image in images]
            patch_mask = [mask.cuda(non_blocking=True).float() for mask in patch_mask]

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student((images[0], images[1], patch_mask[0], patch_mask[1]))
                teacher_out = student_out.detach()
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = -torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_loss = SupConLoss()(student_proj, labels=class_labels[mask_lab])
                loss = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            if batch_idx % args.print_freq == 0:
                args.logger.info(
                    "Epoch: [{}][{}/{}]\t loss {:.5f}\t cls_loss: {:.4f} cluster_loss: {:.4f} "
                    "sup_con_loss: {:.4f} contrastive_loss: {:.4f}".format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        loss.item(),
                        cls_loss.item(),
                        cluster_loss.item(),
                        sup_con_loss.item(),
                        contrastive_loss.item(),
                    )
                )

        args.logger.info("Train Epoch: {} Avg Loss: {:.4f} ".format(epoch, loss_record.avg))
        all_acc, old_acc, new_acc = test(
            student,
            unlabelled_train_loader,
            epoch=epoch,
            save_name="Train ACC Unlabelled",
            args=args,
        )
        args.logger.info("Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(all_acc, old_acc, new_acc))
        exp_lr_scheduler.step()
        torch.save(
            {"model": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1},
            args.model_path,
        )
        args.logger.info("model saved to {}.".format(args.model_path))


def parse_args():
    parser = argparse.ArgumentParser(description="cluster", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--eval_funcs", nargs="+", default=["v2", "v2p"])
    parser.add_argument("--warmup_model_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="scars", help="options: cub, scars, aircraft")
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
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--mask_root", required=True, type=str)
    parser.add_argument("--fusion_heads", default=12, type=int)
    parser.add_argument("--min_foreground_tokens", default=1, type=int)
    parser.add_argument("--max_foreground_tokens", default=None, type=int)
    return parser.parse_args()


def main():
    args = get_class_splits(parse_args())
    device = torch.device("cuda:0")
    set_strict_seed(args.seed)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    init_experiment(args, runner_name=["simgcd-asymmetric-b2-repro"])
    args.logger.info(f"Using evaluation function {args.eval_funcs[0]} to print results")
    args.logger.info(
        "Strict reproducibility enabled | seed=%d | torch=%s | cuda=%s | cudnn=%s | gpu=%s",
        args.seed,
        torch.__version__,
        torch.version.cuda,
        torch.backends.cudnn.version(),
        torch.cuda.get_device_name(device),
    )
    args.interpolation = 3
    args.crop_pct = 0.875
    backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    if args.warmup_model_dir is not None:
        args.logger.info(f"Loading weights from {args.warmup_model_dir}")
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location="cpu"))
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    args.patch_grid_size = int(args.image_size / 16)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for name, parameter in backbone.named_parameters():
        if "block" in name and int(name.split(".")[1]) >= args.grad_from_block:
            parameter.requires_grad = True

    train_transform, test_transform = get_paired_mask_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = PairedMaskViewGenerator(base_transform=train_transform, n_views=args.n_views)
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = build_mask_datasets(
        train_transform, test_transform, args
    )
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = torch.DoubleTensor(
        [1 if index < label_len else label_len / unlabelled_len for index in range(len(train_dataset))]
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader_generator = torch.Generator().manual_seed(args.seed + 1)
    eval_loader_generator = torch.Generator().manual_seed(args.seed + 2)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset), generator=train_generator
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=partial(collate_train_mask_batch, target_grid_size=args.patch_grid_size),
        worker_init_fn=seed_worker,
        generator=train_loader_generator,
    )
    test_loader_unlabelled = DataLoader(
        unlabelled_train_examples_test,
        num_workers=args.num_workers,
        batch_size=256,
        shuffle=False,
        pin_memory=False,
        collate_fn=partial(collate_eval_mask_batch, target_grid_size=args.patch_grid_size),
        worker_init_fn=seed_worker,
        generator=eval_loader_generator,
    )
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = B2ForegroundPairMaskModel(
        backbone=backbone,
        head=projector,
        feat_dim=args.feat_dim,
        fusion_heads=args.fusion_heads,
        min_foreground_tokens=args.min_foreground_tokens,
        max_foreground_tokens=args.max_foreground_tokens,
    ).to(device)
    train(model, train_loader, None, test_loader_unlabelled, args)


if __name__ == "__main__":
    main()
