import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.data_utils import MergedDataset


def infer_image_path(dataset, index):
    if isinstance(dataset, MergedDataset):
        if index < len(dataset.labelled_dataset):
            return infer_image_path(dataset.labelled_dataset, index)
        return infer_image_path(dataset.unlabelled_dataset, index - len(dataset.labelled_dataset))

    if hasattr(dataset, "data"):
        data = dataset.data
        if hasattr(data, "iloc"):
            row = data.iloc[index]
            if hasattr(row, "filepath") and hasattr(dataset, "root") and hasattr(dataset, "base_folder"):
                return os.path.join(dataset.root, dataset.base_folder, row.filepath)
        if isinstance(data, list):
            return data[index]

    if hasattr(dataset, "samples"):
        return dataset.samples[index][0]
    if hasattr(dataset, "imgs"):
        return dataset.imgs[index][0]

    raise RuntimeError(f"Cannot infer image path for dataset type {type(dataset).__name__}.")


def infer_image_root(dataset):
    if isinstance(dataset, MergedDataset):
        return infer_image_root(dataset.labelled_dataset)

    if hasattr(dataset, "root") and hasattr(dataset, "base_folder"):
        return os.path.join(dataset.root, dataset.base_folder)
    if hasattr(dataset, "data_dir"):
        return dataset.data_dir
    if hasattr(dataset, "root"):
        return dataset.root

    raise RuntimeError(f"Cannot infer image root for dataset type {type(dataset).__name__}.")


def mask_path_from_image_path(image_path, image_root, mask_root):
    rel_path = os.path.relpath(image_path, image_root)
    rel_no_ext = os.path.splitext(rel_path)[0]
    return os.path.join(mask_root, rel_no_ext + ".npy")


class MaskedDataset(Dataset):
    def __init__(self, dataset, mask_root, image_root=None, mask_size=224, threshold=0.5, missing="error"):
        self.dataset = dataset
        self.mask_root = os.path.abspath(os.path.expanduser(mask_root))
        self.image_root = os.path.abspath(os.path.expanduser(image_root)) if image_root else infer_image_root(dataset)
        self.mask_size = int(mask_size)
        self.threshold = float(threshold)
        self.missing = missing

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def _load_mask(self, index):
        image_path = infer_image_path(self.dataset, index)
        mask_path = mask_path_from_image_path(image_path, self.image_root, self.mask_root)
        if not os.path.isfile(mask_path):
            if self.missing == "zeros":
                return torch.zeros(self.mask_size, self.mask_size, dtype=torch.float32)
            if self.missing == "ones":
                return torch.ones(self.mask_size, self.mask_size, dtype=torch.float32)
            raise FileNotFoundError(f"Mask file not found for image {image_path}: {mask_path}")

        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask at {mask_path}, got shape {tuple(mask.shape)}")
        mask = (mask > self.threshold).float()
        mask = F.interpolate(
            mask.view(1, 1, *mask.shape),
            size=(self.mask_size, self.mask_size),
            mode="nearest",
        ).view(self.mask_size, self.mask_size)
        return mask

    def __getitem__(self, index):
        item = self.dataset[index]
        image, target, uq_idx = item
        mask = self._load_mask(index)
        return (image, mask), target, uq_idx


class MaskedModelWithHead(torch.nn.Module):
    def __init__(self, backbone, projector):
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x, masks=None):
        features = self.backbone(x, masks=masks)
        return self.projector(features)


def split_masked_batch(batch):
    images, class_labels, uq_idxs, mask_lab = batch
    masks = None
    if (
        isinstance(images, (tuple, list))
        and len(images) == 2
        and isinstance(images[0], (tuple, list))
        and torch.is_tensor(images[1])
    ):
        images, masks = images
    return images, masks, class_labels, uq_idxs, mask_lab


def masks_for_views(masks, images, device):
    if masks is None:
        return None
    masks = masks.to(device, non_blocking=True)
    if isinstance(images, (tuple, list)):
        return torch.cat([masks for _ in images], dim=0)
    return masks
