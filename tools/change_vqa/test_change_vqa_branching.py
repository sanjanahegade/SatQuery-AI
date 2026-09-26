import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from change_vqa_cli import resolve_backend, OPTICAL_PYTHON, SAR_PYTHON

DATA_DIR = r"D:\satQai\data\real_test"
OPT_PRE = os.path.join(DATA_DIR, "sentinel2_mysuru_earlier.tif")
OPT_POST = os.path.join(DATA_DIR, "sentinel2_mysuru.tif")
SAR_PRE = os.path.join(DATA_DIR, "sentinel1_nepal_pre_flood_vv.tif")
SAR_POST = os.path.join(DATA_DIR, "sentinel1_nepal_post_flood_vv.tif")


class TestChangeVQABranching(unittest.TestCase):

    def test_optical_pair_branching(self):
        """Optical + Optical -> modality='optical', backend='GeoChat', exec=OPTICAL_PYTHON"""
        mod, backend, py_exec = resolve_backend(OPT_PRE, OPT_POST)
        self.assertEqual(mod, "optical")
        self.assertEqual(backend, "GeoChat")
        self.assertEqual(os.path.normcase(py_exec), os.path.normcase(OPTICAL_PYTHON))
        self.assertNotEqual(os.path.normcase(py_exec), os.path.normcase(SAR_PYTHON))

    def test_sar_pair_branching(self):
        """SAR + SAR -> modality='sar', backend='Qwen2-VL', exec=SAR_PYTHON"""
        mod, backend, py_exec = resolve_backend(SAR_PRE, SAR_POST)
        self.assertEqual(mod, "sar")
        self.assertEqual(backend, "Qwen2-VL")
        self.assertEqual(os.path.normcase(py_exec), os.path.normcase(SAR_PYTHON))
        self.assertNotEqual(os.path.normcase(py_exec), os.path.normcase(OPTICAL_PYTHON))

    def test_heterogeneous_pair_rejection(self):
        """Optical + SAR -> ValueError rejection (requires matched modality)"""
        with self.assertRaises(ValueError) as ctx:
            resolve_backend(OPT_PRE, SAR_POST)
        self.assertIn("Heterogeneous optical + SAR pair detected", str(ctx.exception))

    def test_explicit_modality_override(self):
        """Explicit --modality flag overrides automatic detection"""
        mod_opt, backend_opt, exec_opt = resolve_backend(SAR_PRE, SAR_POST, explicit_modality="optical")
        self.assertEqual(mod_opt, "optical")
        self.assertEqual(backend_opt, "GeoChat")
        self.assertEqual(os.path.normcase(exec_opt), os.path.normcase(OPTICAL_PYTHON))

        mod_sar, backend_sar, exec_sar = resolve_backend(OPT_PRE, OPT_POST, explicit_modality="sar")
        self.assertEqual(mod_sar, "sar")
        self.assertEqual(backend_sar, "Qwen2-VL")
        self.assertEqual(os.path.normcase(exec_sar), os.path.normcase(SAR_PYTHON))


if __name__ == "__main__":
    unittest.main(verbosity=2)
