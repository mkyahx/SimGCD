"""Summarize foreground occupancy for `.npy` patch masks.

The foreground rule matches `MaskForegroundGLSimModel._align_patch_mask`:
values strictly greater than 0.5 are treated as selected foreground patches.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def compute_foreground_ratios(mask_root):
    """Return per-file foreground ratios and aggregate statistics for a mask root."""
    mask_root = Path(mask_root)
    mask_paths = sorted(mask_root.rglob("*.npy"))
    if not mask_paths:
        raise ValueError(f"No .npy mask files found under: {mask_root}")

    rows = []
    for mask_path in mask_paths:
        mask = np.load(mask_path)
        if mask.size == 0:
            raise ValueError(f"Empty mask file: {mask_path}")
        foreground_ratio = float((mask > 0.5).mean())
        rows.append(
            {
                "path": mask_path.relative_to(mask_root).as_posix(),
                "foreground_ratio": foreground_ratio,
            }
        )

    ratios = np.asarray([row["foreground_ratio"] for row in rows], dtype=np.float64)
    summary = {
        "count": int(ratios.size),
        "mean": float(ratios.mean()),
        "std": float(ratios.std()),
        "median": float(np.median(ratios)),
        "p05": float(np.percentile(ratios, 5)),
        "p95": float(np.percentile(ratios, 95)),
    }
    return rows, summary


def write_csv(rows, csv_path):
    """Write one row per mask for downstream TokenCut/IS-Net comparisons."""
    with Path(csv_path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["path", "foreground_ratio"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute foreground occupancy of .npy masks.")
    parser.add_argument("mask_root", type=Path, help="Root directory containing .npy masks.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional per-mask CSV output path.")
    args = parser.parse_args()

    rows, summary = compute_foreground_ratios(args.mask_root)
    print(f"mask_root: {args.mask_root}")
    for name, value in summary.items():
        print(f"{name}: {value}" if name == "count" else f"{name}: {value:.6f}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.csv)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
