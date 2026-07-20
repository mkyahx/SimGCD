import argparse
import math
import os
import random
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import exp_root
from data.get_datasets import get_class_splits, get_dataset_funcs
from data.mask_dataset import DatasetWithPatchMask, MergedDatasetMask
from data.paired_mask_transforms import PairedMaskViewGenerator, get_paired_mask_transform
from glsim_mask_model import MaskForegroundGLSimModel
from model import DINOHead, DistillLoss, SupConLoss, get_params_groups, info_nce_logits
from util.cluster_and_log_utils import log_accs_from_preds
from util.general_utils import AverageMeter, init_experiment


def _stack_image_field(image_batch):
    if isinstance(image_batch[0], (list, tuple)):
        num_views = len(image_batch[0])
        return [
            torch.stack([sample[view_idx].clone().contiguous() for sample in image_batch], dim=0)
            for view_idx in range(num_views)
        ]
    return torch.stack([sample.clone().contiguous() for sample in image_batch], dim=0)


def _resize_patch_mask(sample, target_grid_size):
    sample = sample.clone().contiguous().float()
    if sample.dim() == 2:
        sample = sample.unsqueeze(0).unsqueeze(0)
    elif sample.dim() == 3:
        sample = sample.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported patch mask shape: {tuple(sample.shape)}")

    sample = F.interpolate(sample, size=(target_grid_size, target_grid_size), mode="nearest")
    sample = (sample > 0.5).float()
    return sample.squeeze(0).squeeze(0)


def _stack_patch_mask_field(mask_batch, target_grid_size):
    if isinstance(mask_batch[0], (list, tuple)):
        num_views = len(mask_batch[0])
        return [
            torch.stack(
                [_resize_patch_mask(sample[view_idx], target_grid_size) for sample in mask_batch],
                dim=0,
            )
            for view_idx in range(num_views)
        ]
    return torch.stack(
        [_resize_patch_mask(sample, target_grid_size) for sample in mask_batch],
        dim=0,
    )


def collate_train_mask_batch(batch, target_grid_size):
    images, class_labels, uq_idxs, mask_lab, patch_masks = zip(*batch)
    images = _stack_image_field(images)
    class_labels = torch.as_tensor(class_labels)
    uq_idxs = torch.as_tensor(uq_idxs)
    mask_lab = torch.stack([torch.as_tensor(sample).clone().contiguous() for sample in mask_lab], dim=0)
    patch_masks = _stack_patch_mask_field(patch_masks, target_grid_size)
    return images, class_labels, uq_idxs, mask_lab, patch_masks


def collate_eval_mask_batch(batch, target_grid_size):
    images, class_labels, uq_idxs, patch_masks = zip(*batch)
    images = _stack_image_field(images)
    class_labels = torch.as_tensor(class_labels)
    uq_idxs = torch.as_tensor(uq_idxs)
    patch_masks = torch.stack(
        [_resize_patch_mask(sample, target_grid_size) for sample in patch_masks],
        dim=0,
    )
    return images, class_labels, uq_idxs, patch_masks


def build_mask_datasets(train_transform, test_transform, args):
    supported_mask_datasets = {"cub", "aircraft", "scars"}
    if args.dataset_name not in supported_mask_datasets:
        raise NotImplementedError(
            f"The GLSim mask entrypoint supports {sorted(supported_mask_datasets)}, "
            f"got {args.dataset_name!r}."
        )

    get_dataset_f = get_dataset_funcs[args.dataset_name]
    datasets = get_dataset_f(
        train_transform=None,
        test_transform=None,
        train_classes=args.train_classes,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
    )

    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset in datasets.values():
        if dataset is not None:
            dataset.target_transform = target_transform

    train_dataset = MergedDatasetMask(
        labelled_dataset=DatasetWithPatchMask(
            deepcopy(datasets["train_labelled"]),
            mask_root=args.mask_root,
            transform=train_transform,
        ),
        unlabelled_dataset=DatasetWithPatchMask(
            deepcopy(datasets["train_unlabelled"]),
            mask_root=args.mask_root,
            transform=train_transform,
        ),
    )
    test_dataset = DatasetWithPatchMask(
        datasets["test"],
        mask_root=args.mask_root,
        transform=test_transform,
    )
    unlabelled_train_examples_test = DatasetWithPatchMask(
        deepcopy(datasets["train_unlabelled"]),
        mask_root=args.mask_root,
        transform=test_transform,
    )
    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

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
            images = torch.cat(images, dim=0).cuda(non_blocking=True)
            patch_mask = torch.cat(patch_mask, dim=0).cuda(non_blocking=True).float()

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student((images, patch_mask))
                teacher_out = student_out.detach()

                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ""
                pstr += f"cls_loss: {cls_loss.item():.4f} "
                pstr += f"cluster_loss: {cluster_loss.item():.4f} "
                pstr += f"sup_con_loss: {sup_con_loss.item():.4f} "
                pstr += f"contrastive_loss: {contrastive_loss.item():.4f} "

                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
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
                    "Epoch: [{}][{}/{}]\t loss {:.5f}\t {}".format(
                        epoch, batch_idx, len(train_loader), loss.item(), pstr
                    )
                )

        args.logger.info("Train Epoch: {} Avg Loss: {:.4f} ".format(epoch, loss_record.avg))

        args.logger.info("Testing on unlabelled examples in the training data...")
        all_acc, old_acc, new_acc = test(
            student,
            unlabelled_train_loader,
            epoch=epoch,
            save_name="Train ACC Unlabelled",
            args=args,
        )

        args.logger.info("Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(all_acc, old_acc, new_acc))

        exp_lr_scheduler.step()

        save_dict = {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))


def test(model, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        images, label, _, patch_mask = batch
        images = images.cuda(non_blocking=True)
        patch_mask = patch_mask.cuda(non_blocking=True).float()
        with torch.no_grad():
            _, logits = model((images, patch_mask))
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets,
        y_pred=preds,
        mask=mask,
        T=epoch,
        eval_funcs=args.eval_funcs,
        save_name=save_name,
        args=args,
    )

    return all_acc, old_acc, new_acc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--eval_funcs", nargs="+", help="Which eval functions to use", default=["v2", "v2p"])

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
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--mask_root', required=True, type=str)
    parser.add_argument("--fusion_heads", default=12, type=int)
    parser.add_argument("--min_foreground_tokens", default=1, type=int)

    args = parser.parse_args()
    device = torch.device("cuda:0")
    args = get_class_splits(args)

    if args.seed is not None:
        set_seed(args.seed)
    else:
        torch.backends.cudnn.benchmark = True

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=["simgcd-glsim-mask"])
    args.logger.info(f"Using evaluation function {args.eval_funcs[0]} to print results")

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

    for m in backbone.parameters():
        m.requires_grad = False

    for name, m in backbone.named_parameters():
        if "block" in name:
            block_num = int(name.split(".")[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info("model build")

    train_transform, test_transform = get_paired_mask_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = PairedMaskViewGenerator(base_transform=train_transform, n_views=args.n_views)

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = build_mask_datasets(
        train_transform,
        test_transform,
        args,
    )

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_collate_fn = partial(collate_train_mask_batch, target_grid_size=args.patch_grid_size)
    eval_collate_fn = partial(collate_eval_mask_batch, target_grid_size=args.patch_grid_size)

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=train_collate_fn,
    )
    test_loader_unlabelled = DataLoader(
        unlabelled_train_examples_test,
        num_workers=args.num_workers,
        batch_size=256,
        shuffle=False,
        pin_memory=False,
        collate_fn=eval_collate_fn,
    )

    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = MaskForegroundGLSimModel(
        backbone=backbone,
        head=projector,
        feat_dim=args.feat_dim,
        fusion_heads=args.fusion_heads,
        min_foreground_tokens=args.min_foreground_tokens,
    ).to(device)

    train(model, train_loader, None, test_loader_unlabelled, args)
