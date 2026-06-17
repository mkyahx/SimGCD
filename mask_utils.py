import os

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
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


def path_is_under(path, root):
    path = os.path.abspath(os.path.expanduser(path))
    root = os.path.abspath(os.path.expanduser(root))
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def mask_path_from_image_path(image_path, image_root, mask_root, fallback_image_root=None):
    image_path = os.path.abspath(os.path.expanduser(image_path))
    candidate_roots = []
    for root in (image_root, fallback_image_root):
        if root:
            root = os.path.abspath(os.path.expanduser(root))
            if root not in candidate_roots:
                candidate_roots.append(root)

    chosen_root = None
    for root in candidate_roots:
        if path_is_under(image_path, root):
            chosen_root = root
            break

    if chosen_root is None:
        raise ValueError(
            f"Image path {image_path} is not under --mask_image_root {image_root} "
            f"or inferred image root {fallback_image_root}. "
            "Set --mask_image_root to the image directory whose internal layout matches --mask_root, "
            "or omit it to use the inferred dataset image root."
        )

    rel_path = os.path.relpath(image_path, chosen_root)
    rel_no_ext = os.path.splitext(rel_path)[0]
    return os.path.join(mask_root, rel_no_ext + ".npy")


class MaskedDataset(Dataset):
    def __init__(
        self,
        dataset,
        mask_root,
        image_root=None,
        mask_size=224,
        threshold=0.5,
        missing="error",
        transform=None,
    ):
        self.dataset = dataset
        self.mask_root = os.path.abspath(os.path.expanduser(mask_root))
        self.inferred_image_root = os.path.abspath(os.path.expanduser(infer_image_root(dataset)))
        self.image_root = os.path.abspath(os.path.expanduser(image_root)) if image_root else self.inferred_image_root
        self.mask_size = int(mask_size)
        self.threshold = float(threshold)
        self.missing = missing
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def _load_mask(self, index):
        image_path = infer_image_path(self.dataset, index)
        mask_path = mask_path_from_image_path(
            image_path,
            self.image_root,
            self.mask_root,
            fallback_image_root=self.inferred_image_root,
        )
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
        return (mask > self.threshold).float()

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item[0]
        rest = item[1:]
        mask = self._load_mask(index)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return ((image, mask), *rest)


class PairedImagenetTransform:
    def __init__(self, image_size, interpolation, crop_pct, train=True):
        self.image_size = int(image_size)
        self.resize_size = int(image_size / crop_pct)
        interpolation_map = {
            0: InterpolationMode.NEAREST,
            2: InterpolationMode.BILINEAR,
            3: InterpolationMode.BICUBIC,
        }
        self.interpolation = interpolation_map.get(interpolation, interpolation)
        self.train = train
        self.color_jitter = transforms.ColorJitter()
        self.mean = torch.tensor((0.485, 0.456, 0.406))
        self.std = torch.tensor((0.229, 0.224, 0.225))

    def _resize(self, image, mask):
        image = TF.resize(image, self.resize_size, interpolation=self.interpolation)
        mask = TF.resize(mask.unsqueeze(0), self.resize_size, interpolation=InterpolationMode.NEAREST).squeeze(0)
        return image, mask

    def _to_tensor(self, image, mask):
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        return image, (mask > 0.5).float()

    def __call__(self, image, mask):
        image, mask = self._resize(image, mask)
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(image, (self.image_size, self.image_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            image = self.color_jitter(image)
        else:
            image = TF.center_crop(image, self.image_size)
            mask = TF.center_crop(mask, self.image_size)
        return self._to_tensor(image, mask)


class MaskedContrastiveLearningViewGenerator:
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, image, mask):
        images, masks = [], []
        for _ in range(self.n_views):
            image_view, mask_view = self.base_transform(image, mask)
            images.append(image_view)
            masks.append(mask_view)
        return images, masks


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
        and isinstance(images[1], (tuple, list))
    ):
        images, masks = images
    return images, masks, class_labels, uq_idxs, mask_lab


def masks_for_views(masks, images, device):
    if masks is None:
        return None
    if isinstance(images, (tuple, list)):
        return torch.cat(masks, dim=0).to(device, non_blocking=True)
    return masks.to(device, non_blocking=True)


def clear_image_transform(dataset):
    if isinstance(dataset, MergedDataset):
        clear_image_transform(dataset.labelled_dataset)
        clear_image_transform(dataset.unlabelled_dataset)
        return
    if hasattr(dataset, "transform"):
        dataset.transform = None
