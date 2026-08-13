"""Dilate or erode binary masks on SimGCD's 14 x 14 token grid.

Input masks may have any two-dimensional resolution.  They are first reduced
with nearest-neighbour sampling to the token grid used by SimGCD, then a
four-connected binary morphology operation is applied.  The result is saved
as a new 14 x 14 float32 `.npy` mask tree, leaving source masks untouched.
"""

import argparse
from pathlib import Path

import numpy as np


def resize_mask_to_token_grid(mask, grid_size=14):
    """Convert a 2D mask to a binary square token grid using nearest sampling."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")
    if grid_size < 1:
        raise ValueError("grid_size must be positive.")

    binary_mask = mask > 0.5
    height, width = binary_mask.shape
    row_indices = np.arange(grid_size) * height // grid_size
    col_indices = np.arange(grid_size) * width // grid_size
    return binary_mask[np.ix_(row_indices, col_indices)]


def _dilate_once(mask):
    """Expand foreground by one four-connected token ring."""
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _erode_once(mask):
    """Remove one four-connected outer token ring from foreground."""
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def apply_morphology(mask, mode, iterations):
    """Apply `iterations` four-connected dilations or erosions to a binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")
    if mode not in {"dilate", "erode"}:
        raise ValueError("mode must be either 'dilate' or 'erode'.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")

    result = mask.astype(bool, copy=True)
    operation = _dilate_once if mode == "dilate" else _erode_once
    for _ in range(iterations):
        result = operation(result)
    return result


def process_mask_root(source_root, output_root, mode, iterations, grid_size=14):
    """Write morphed masks to an independent mirrored output directory tree."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("output_root must differ from source_root to preserve source masks.")

    mask_paths = sorted(source_root.rglob("*.npy"))
    if not mask_paths:
        raise ValueError(f"No .npy mask files found under: {source_root}")

    for source_path in mask_paths:
        mask = np.load(source_path, allow_pickle=False)
        token_mask = resize_mask_to_token_grid(mask, grid_size=grid_size)
        result = apply_morphology(token_mask, mode=mode, iterations=iterations)
        output_path = output_root / source_path.relative_to(source_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result.astype(np.float32))

    return len(mask_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Dilate or erode .npy masks on the SimGCD token grid."
    )
    parser.add_argument("source_root", type=Path, help="Input directory of .npy masks.")
    parser.add_argument("output_root", type=Path, help="New output directory for morphed masks.")
    parser.add_argument("--mode", required=True, choices=("dilate", "erode"))
    parser.add_argument("--iterations", type=int, default=1, help="Number of token rings to add or remove.")
    parser.add_argument("--grid-size", type=int, default=14, help="Square output token-grid size.")
    args = parser.parse_args()

    count = process_mask_root(
        args.source_root,
        args.output_root,
        mode=args.mode,
        iterations=args.iterations,
        grid_size=args.grid_size,
    )
    print(f"source_root: {args.source_root}")
    print(f"output_root: {args.output_root}")
    print(f"mode: {args.mode}")
    print(f"iterations: {args.iterations}")
    print(f"grid_size: {args.grid_size}")
    print(f"processed_count: {count}")


if __name__ == "__main__":
    main()
