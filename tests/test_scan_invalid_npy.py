import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "scan_invalid_npy.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("scan_invalid_npy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScanInvalidNpyTests(unittest.TestCase):
    def test_recursively_reports_only_unloadable_npy_files(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "valid.npy", np.ones((2, 2), dtype=np.float32))
            nested = root / "nested"
            nested.mkdir()
            broken_path = nested / "broken.npy"
            broken_path.write_bytes(b"not-a-numpy-file")

            invalid_files = module.scan_invalid_npy_files(root)

        self.assertEqual([path.name for path, _ in invalid_files], ["broken.npy"])
        self.assertTrue(invalid_files[0][1])

    def test_empty_root_has_no_invalid_files(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(module.scan_invalid_npy_files(Path(temp_dir)), [])


if __name__ == "__main__":
    unittest.main()
