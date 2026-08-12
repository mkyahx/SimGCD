"""Find unreadable `.npy` files under a directory without modifying them."""

import argparse
from pathlib import Path

import numpy as np


def scan_invalid_npy_files(root):
    """Return `(path, error)` pairs for `.npy` files that NumPy cannot fully load."""
    root = Path(root)
    invalid_files = []
    for npy_path in sorted(root.rglob("*.npy")):
        try:
            # Materializing the array catches truncated payloads whose headers
            # can otherwise be parsed successfully.
            np.load(npy_path, allow_pickle=False)
        except Exception as error:
            invalid_files.append((npy_path, str(error)))
    return invalid_files


def main():
    parser = argparse.ArgumentParser(description="Recursively find unreadable .npy files.")
    parser.add_argument("root", type=Path, help="Directory to scan recursively.")
    args = parser.parse_args()

    invalid_files = scan_invalid_npy_files(args.root)
    print(f"scan_root: {args.root}")
    print(f"invalid_count: {len(invalid_files)}")
    for path, error in invalid_files:
        print(f"INVALID: {path}")
        print(f"  error: {error}")

    if invalid_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
