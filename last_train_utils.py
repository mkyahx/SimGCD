import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from mask_utils import masks_for_views, split_masked_batch
from model import DistillLoss, SupConLoss, get_params_groups, info_nce_logits
from util.cluster_and_log_utils import log_accs_from_preds
from util.general_utils import AverageMeter


def train_with_optional_masks(student, train_loader, test_loader, unlabelled_train_loader, args):
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
            images, masks, class_labels, uq_idxs, mask_lab = split_masked_batch(batch)
            mask_lab = mask_lab[:, 0]

            class_labels = class_labels.cuda(non_blocking=True)
            mask_lab = mask_lab.cuda(non_blocking=True).bool()
            view_masks = masks_for_views(masks, images, torch.device("cuda"))
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images, view_masks)
                teacher_out = student_out.detach()

                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = -torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
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
        all_acc, old_acc, new_acc = test_with_optional_masks(
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


def test_with_optional_masks(model, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        images, masks, label, _ = split_test_batch(batch)
        images = images.cuda(non_blocking=True)
        masks = masks_for_views(masks, images, torch.device("cuda"))
        with torch.no_grad():
            _, logits = model(images, masks)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([x.item() in range(len(args.train_classes)) for x in label]))

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


def split_test_batch(batch):
    images, label, uq_idxs = batch
    masks = None
    if isinstance(images, (tuple, list)) and len(images) == 2 and torch.is_tensor(images[1]):
        images, masks = images
    return images, masks, label, uq_idxs
