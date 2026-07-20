import random

import numpy as np
from PIL import Image, ImageOps


def _to_pil_mask(mask):
    if isinstance(mask, Image.Image):
        return mask.convert("L")

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")

    mask = (mask > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def _resize_short_edge_pair(image, mask, short_edge_size, interpolation):
    width, height = image.size
    if width <= height:
        new_width = short_edge_size
        new_height = int(round(short_edge_size * height / width))
    else:
        new_height = short_edge_size
        new_width = int(round(short_edge_size * width / height))

    image = image.resize((new_width, new_height), resample=interpolation)
    mask = mask.resize((new_width, new_height), resample=Image.Resampling.NEAREST)
    return image, mask


def crop_pair(image, mask, top, left, size):
    box = (left, top, left + size, top + size)
    return image.crop(box), mask.crop(box)


def center_crop_pair(image, mask, size):
    width, height = image.size
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return crop_pair(image, mask, top=top, left=left, size=size)


def random_crop_pair(image, mask, size):
    width, height = image.size
    if width == size and height == size:
        return image, mask
    if width < size or height < size:
        raise ValueError(f"Cannot crop size {size} from image size {(width, height)}.")

    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    return crop_pair(image, mask, top=top, left=left, size=size)


def hflip_pair(image, mask):
    return ImageOps.mirror(image), ImageOps.mirror(mask)


def _image_to_normalized_tensor(image, mean, std):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def _mask_to_patch_tensor(mask, patch_grid_size):
    import torch

    mask = mask.resize((patch_grid_size, patch_grid_size), resample=Image.Resampling.NEAREST)
    array = (np.asarray(mask, dtype=np.float32) > 127.5).astype(np.float32)
    return torch.from_numpy(array)


class PairedImageMaskTransform:
    supports_mask_pair = True

    def __init__(
        self,
        image_size,
        crop_pct,
        interpolation,
        patch_grid_size,
        train,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        hflip_prob=0.5,
    ):
        self.image_size = image_size
        self.resize_size = int(image_size / crop_pct)
        self.interpolation = interpolation
        self.patch_grid_size = patch_grid_size
        self.train = train
        self.mean = mean
        self.std = std
        self.hflip_prob = hflip_prob

    def __call__(self, image, mask):
        mask = _to_pil_mask(mask)
        if mask.size != image.size:
            mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)

        image, mask = _resize_short_edge_pair(
            image,
            mask,
            short_edge_size=self.resize_size,
            interpolation=self.interpolation,
        )

        if self.train:
            image, mask = random_crop_pair(image, mask, self.image_size)
            if random.random() < self.hflip_prob:
                image, mask = hflip_pair(image, mask)
        else:
            image, mask = center_crop_pair(image, mask, self.image_size)

        return (
            _image_to_normalized_tensor(image, self.mean, self.std),
            _mask_to_patch_tensor(mask, self.patch_grid_size),
        )


class PairedMaskViewGenerator:
    supports_mask_pair = True

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, image, mask):
        images = []
        masks = []
        for _ in range(self.n_views):
            image_view, mask_view = self.base_transform(image, mask)
            images.append(image_view)
            masks.append(mask_view)
        return images, masks


def get_paired_mask_transform(transform_type, image_size, args):
    if transform_type != "imagenet":
        raise NotImplementedError

    patch_grid_size = int(image_size / 16) if not isinstance(image_size, tuple) else int(image_size[0] / 16)
    train_transform = PairedImageMaskTransform(
        image_size=image_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        patch_grid_size=patch_grid_size,
        train=True,
    )
    test_transform = PairedImageMaskTransform(
        image_size=image_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        patch_grid_size=patch_grid_size,
        train=False,
    )
    return train_transform, test_transform
