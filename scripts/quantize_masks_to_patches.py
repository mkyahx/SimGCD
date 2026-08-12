"""Convert pixel masks into piecewise-constant patch-aligned masks.

Each `patch_size` × `patch_size` block is foreground when at least half of
its source pixels are foreground (`mask > 0.5`).  Outputs retain the source
mask resolution and relative directory structure, but each block is all 0 or
all 1 so the mask has no sub-patch detail.
"""

import argparse
from pathlib import Path

import numpy as np


def quantize_mask_to_patches(mask, patch_size=16):
    """Return a same-shape binary mask quantized to patch-aligned blocks."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")
    if patch_size < 1:
        raise ValueError("patch_size must be positive.")

    binary_mask = mask > 0.5
    height, width = binary_mask.shape
    quantized = np.zeros((height, width), dtype=np.float32)
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            block = binary_mask[top : top + patch_size, left : left + patch_size]
            quantized[top : top + patch_size, left : left + patch_size] = float(
                block.mean() >= 0.5
            )
    return quantized


def process_mask_root(source_root, output_root, patch_size=16):
    """Quantize every `.npy` mask into an independent mirrored output tree."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("output_root must differ from source_root to preserve source masks.")

    mask_paths = sorted(source_root.rglob("*.npy"))
    if not mask_paths:
        raise ValueError(f"No .npy mask files found under: {source_root}")

    for source_path in mask_paths:
        mask = np.load(source_path, allow_pickle=False)
        quantized = quantize_mask_to_patches(mask, patch_size=patch_size)
        output_path = output_root / source_path.relative_to(source_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, quantized)

    return len(mask_paths)


def main():
    parser = argparse.ArgumentParser(description="Quantize .npy masks to patch-aligned blocks.")
    parser.add_argument("source_root", type=Path, help="Input directory of pixel-level .npy masks.")
    parser.add_argument("output_root", type=Path, help="New output directory for quantized .npy masks.")
    parser.add_argument("--patch-size", default=16, type=int, help="Pixel width/height of one block.")
    args = parser.parse_args()

    count = process_mask_root(args.source_root, args.output_root, patch_size=args.patch_size)
    print(f"source_root: {args.source_root}")
    print(f"output_root: {args.output_root}")
    print(f"patch_size: {args.patch_size}")
    print(f"processed_count: {count}")


if __name__ == "__main__":
    main()
