import os

import numpy as np
from torch.utils.data import Dataset


def _without_extension(path):
    return os.path.splitext(path)[0]


def _unique_in_order(items):
    seen = set()
    unique = []
    for item in items:
        norm = item.replace("\\", os.sep).replace("/", os.sep)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def default_image_path_getter(dataset, idx):
    if hasattr(dataset, "data"):
        data = dataset.data
        if hasattr(data, "iloc") and "filepath" in data.columns:
            return data.iloc[idx].filepath
        if isinstance(data, (list, tuple, np.ndarray)):
            return data[idx]

    if hasattr(dataset, "samples"):
        return dataset.samples[idx][0]

    raise AttributeError(
        f"Cannot infer image path for {type(dataset).__name__}; pass an explicit path_getter."
    )


def candidate_mask_relpaths(image_path):
    image_path = os.fspath(image_path)
    normalized = image_path.replace("\\", os.sep).replace("/", os.sep)
    no_ext = _without_extension(normalized)
    parts = [part for part in no_ext.split(os.sep) if part]

    candidates = []
    if not os.path.isabs(normalized):
        candidates.append(no_ext)
    if parts:
        candidates.append(parts[-1])

    for tail_len in (2, 3):
        if len(parts) >= tail_len:
            candidates.append(os.path.join(*parts[-tail_len:]))

    for anchor in ("images", "cars_train", "cars_test"):
        if anchor in parts:
            anchor_index = parts.index(anchor)
            candidates.append(os.path.join(*parts[anchor_index:]))
            if anchor_index + 1 < len(parts):
                candidates.append(os.path.join(*parts[anchor_index + 1:]))

    return [relpath + ".npy" for relpath in _unique_in_order(candidates)]


class DatasetWithPatchMask(Dataset):
    def __init__(self, dataset, mask_root=None, transform=None, path_getter=default_image_path_getter):
        self.dataset = dataset
        self.mask_root = os.path.expanduser(mask_root) if mask_root is not None else None
        self.transform = transform
        self.path_getter = path_getter
        self.target_transform = getattr(dataset, "target_transform", None)

    def __len__(self):
        return len(self.dataset)

    def _get_mask(self, idx):
        if self.mask_root is None:
            return np.ones((14, 14), dtype=np.float32)

        image_path = self.path_getter(self.dataset, idx)
        attempted_paths = []
        for relpath in candidate_mask_relpaths(image_path):
            mask_path = os.path.join(self.mask_root, relpath)
            attempted_paths.append(mask_path)
            if os.path.isfile(mask_path):
                return np.load(mask_path).astype(np.float32)

        attempted = "\n  ".join(attempted_paths)
        raise FileNotFoundError(f"Mask file not found for image {image_path}. Tried:\n  {attempted}")

    def __getitem__(self, idx):
        import torch

        img, target, uq_idx = self.dataset[idx]
        patch_mask = torch.tensor(self._get_mask(idx).copy(), dtype=torch.float32)

        if self.transform is not None:
            if getattr(self.transform, "supports_mask_pair", False):
                img, patch_mask = self.transform(img, patch_mask)
            else:
                img = self.transform(img)

        return img, target, uq_idx, patch_mask


class MergedDatasetMask(Dataset):
    def __init__(self, labelled_dataset, unlabelled_dataset):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            img, label, uq_idx, patch_mask = self.labelled_dataset[item]
            labeled_or_not = 1
        else:
            img, label, uq_idx, patch_mask = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        return img, label, uq_idx, np.array([labeled_or_not]), patch_mask

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
