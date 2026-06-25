"""
MTMC: Maximum Token Manifold Capacity for Generalized Category Discovery.

Reference:
  Luyao Tang, Kunze Huang, Chaoqi Chen, Cheng Chen.
  "Generalized Category Discovery via Token Manifold Capacity Learning."
  arXiv:2505.14044v1

Paper-faithful implementation:

  Backbone ........ DINO-pretrained ViT-B/16 (DINOv1, paper default).
                    DINOv2 ViT-B/14 is also supported via --backbone dinov2.

  L_MTMC ......... -sum_r sigma_r([cls]^u) on the unlabelled [CLS] tokens
                    of the mini-batch. Exactly the paper's equation; no
                    row-normalisation, no sqrt(B_u) scaling, no diagonal
                    jitter. (Engineering fallbacks are opt-in via
                    --mtmc_engineered.)

  L_total ........ L_GCD + lambda * L_MTMC, with lambda = --mtmc_weight.

Note: the official MTMC repo (https://github.com/lytang63/MTMC) currently
contains only a README ("Coming soon!"). The implementation here follows the
paper text directly.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import (
    DINOHead,
    SupConLoss,
    DistillLoss,
    ContrastiveLearningViewGenerator,
    info_nce_logits,
    get_params_groups,
)


# ---------------------------------------------------------------------------
# Reproducibility helpers (kept identical to train.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------
def load_backbone(name):
    """Returns a torch.nn.Module whose forward(x) returns cls tokens [B, D].

    Both DINOv1 ViT-B/16 and DINOv2 ViT-B/14 already collapse the cls token
    in their public hub wrappers, so the same forward signature works for
    both backbones. No CLS-slice needed.
    """
    if name == "dinov1":
        return torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    if name == "dinov2":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    raise ValueError(f"Unknown backbone: {name}")


# ---------------------------------------------------------------------------
# MTMC loss (paper-faithful by default)
# ---------------------------------------------------------------------------
def mtmc_loss(cls_tokens, mask_lab, engineered=False):
    """MTMC loss on the unlabelled class tokens.

    Paper-faithful formulation (engineered=False):

        L_MTMC = -||[cls]^u||_* = -sum_r sigma_r([cls]^u)

    where sigma_r are the singular values of the unlabelled [CLS] matrix
    [B_u, D]. This matches Equation 3.2 of arXiv:2505.14044.

    Engineering fallback (engineered=True) keeps the math the same but
    applies L2 row normalisation before SVD and scales by 1/sqrt(B_u).
    Useful when raw DINOv2 [CLS] tokens are too anisotropic for cuSOLVER
    to converge; not used in the paper.

    Args:
        cls_tokens: [B, D] class-token embeddings of the *first* view.
        mask_lab:   [B] bool; True for labelled samples.
        engineered: apply row-L2 + sqrt(B_u) compensation.

    Returns:
        Scalar ``-sum_r sigma_r`` (or with the sqrt(B_u) scale if
        engineered=True). Returns 0 if there are < 2 unlabelled samples
        or if SVD still fails after fp32 casting.
    """
    unlabelled = cls_tokens[~mask_lab]
    if unlabelled.numel() == 0 or unlabelled.size(0) < 2:
        return cls_tokens.new_zeros(())

    # SVD must run in fp32; autocast would otherwise downcast linalg ops
    # to fp16/bf16 and cuSOLVER fails on DINO features.
    unlabelled = unlabelled.float()

    if engineered:
        unlabelled = torch.nn.functional.normalize(unlabelled, dim=-1, eps=1e-6)

    try:
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.linalg.svdvals(unlabelled)
    except Exception:
        # Single-batch SVD failure should not kill the entire run.
        return cls_tokens.new_zeros(())

    nuc_norm = s.sum()
    if engineered:
        nuc_norm = nuc_norm / math.sqrt(float(unlabelled.size(0)))

    return -nuc_norm  # L_MTMC per the paper's equation


# ---------------------------------------------------------------------------
# Train / test (mirrors train.py, only the cluster loss is augmented)
# ---------------------------------------------------------------------------
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

    # --- MTMC: forward hook captures [CLS] tokens from the backbone ---
    backbone = student[0]
    captured = {}

    def _capture_hook(module, inputs, output):
        # Both DINOv1 and DINOv2 hub wrappers return cls tokens [B, D]
        # directly, so the hook captures exactly the [CLS] representation
        # the paper applies MTMC to.
        captured["cls_tokens"] = output

    backbone.register_forward_hook(_capture_hook)

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)
                teacher_out = student_out.detach()
                cls_tokens = captured["cls_tokens"]

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # --- MTMC: maximise nuclear norm of unlabelled cls tokens ---
                first_view_cls = cls_tokens[: args.batch_size]
                mtmc = mtmc_loss(first_view_cls, mask_lab, engineered=args.mtmc_engineered)
                cluster_loss += args.mtmc_weight * mtmc

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'me_max: {me_max_loss.item():.4f} '
                pstr += f'mtmc: {mtmc.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

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
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)

        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))


def test(model, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MTMC: paper-faithful GCD baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument(
        '--backbone', type=str, default='dinov1', choices=['dinov1', 'dinov2'],
        help='dinov1 = DINO ViT-B/16 (paper default). dinov2 = DINOv2 ViT-B/14 '
             '(SimGCD default; produces a different feature distribution).'
    )
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument(
        '--mtmc_weight', type=float, default=0.05,
        help='lambda in L_GCD + lambda * L_MTMC. Paper does not pin a value; '
             'the paper-faithful loss returns -sum sigma_r on raw [CLS] '
             'tokens (norm ~ 60 for DINOv1), so its magnitude is ~300 for a '
             'batch of 16 unlabelled samples. The default 0.05 gives a '
             'contribution ~15 to the total loss, on the same order as '
             'cls_loss (~5) and me_max_loss (~10). Sweep {0.01, 0.05, 0.1} '
             'if you want to match the paper\'s reported +1.4 on CUB.'
    )
    parser.add_argument(
        '--mtmc_engineered', action='store_true', default=False,
        help='Apply L2 row normalisation + 1/sqrt(B_u) scaling inside the MTMC '
             'loss. Engineering fallback; not in the paper. Use this only if '
             'SVD keeps failing on your backbone.'
    )
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--seed', type=int, default=0, help='Set to enable reproducible training.')
    parser.add_argument('--no_seed', action='store_const', const=None, dest='seed', help='Disable explicit random seeding.')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic CUDA/PyTorch behavior when --seed is set.')
    parser.add_argument('--no_deterministic', action='store_false', dest='deterministic', help='Allow non-deterministic CUDA/PyTorch behavior.')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['mtmc'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    args.logger.info(f'Backbone: {args.backbone}  '
                     f'(paper default = dinov1 = DINO ViT-B/16)')
    args.logger.info(f'MTMC weight: {args.mtmc_weight}  '
                     f'engineered compensation: {args.mtmc_engineered}')

    torch.backends.cudnn.benchmark = args.seed is None or not args.deterministic

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = load_backbone(args.backbone)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # --------------------
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

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True,
                              worker_init_fn=seed_worker if args.seed is not None else None,
                              generator=loader_generator)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False,
                                        worker_init_fn=seed_worker if args.seed is not None else None,
                                        generator=loader_generator)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, None, test_loader_unlabelled, args)
