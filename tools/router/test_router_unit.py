import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from router import run_satquery, classify_task, OPTICAL_PYTHON, SAR_PYTHON

# Real files for existence checks
DATA_DIR = r"D:\satQai\data\real_test"
OPTICAL_FILE = os.path.join(DATA_DIR, "sentinel2_mysuru.tif")
OPTICAL_FILE2 = os.path.join(DATA_DIR, "sentinel2_mysuru_earlier.tif")
SAR_FILE = os.path.join(DATA_DIR, "sentinel1_mysuru_registered_vv.tif")
SAR_FILE2 = os.path.join(DATA_DIR, "sentinel1_mysuru_vv.tif")


class TestSatQueryRouter(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(OPTICAL_FILE), f"Missing test file: {OPTICAL_FILE}")
        self.assertTrue(os.path.exists(SAR_FILE), f"Missing test file: {SAR_FILE}")

    def assert_standard_schema(self, res, expect_error=False):
        """Assert presence and types for all 8 standard result schema fields."""
        self.assertIn("task", res)
        self.assertIsInstance(res["task"], str)

        self.assertIn("modalities", res)
        self.assertIsInstance(res["modalities"], list)

        self.assertIn("specialists_used", res)
        self.assertIsInstance(res["specialists_used"], list)

        self.assertIn("answer", res)
        self.assertIsInstance(res["answer"], str)

        self.assertIn("evidence", res)
        self.assertIsInstance(res["evidence"], list)

        self.assertIn("artifacts", res)
        self.assertIsInstance(res["artifacts"], list)

        self.assertIn("metadata", res)
        self.assertIsInstance(res["metadata"], dict)

        self.assertIn("timing", res)
        self.assertIsInstance(res["timing"], dict)
        self.assertIn("router_total_time_s", res["timing"])

        if expect_error:
            self.assertIn("error", res)
            self.assertIsInstance(res["error"], dict)
            self.assertIn("type", res["error"])
            self.assertIn("message", res["error"])

    @patch("router.detect_modality")
    def test_01_single_optical_vqa(self, mock_detect):
        """TEST 1: One optical image, 'What buildings are visible?' -> optical_vqa"""
        mock_detect.return_value = {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical"]}
        res = run_satquery([OPTICAL_FILE], "What buildings are visible?", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "optical_vqa")
        self.assertEqual(res["modalities"][0]["modality"], "optical")

    @patch("router.detect_modality")
    def test_02_single_optical_caption(self, mock_detect):
        """TEST 2: One optical image, 'Describe this image.' -> optical_caption"""
        mock_detect.return_value = {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical"]}
        res = run_satquery([OPTICAL_FILE], "Describe this image.", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "optical_caption")
        self.assertEqual(res["modalities"][0]["modality"], "optical")

    @patch("router.detect_modality")
    def test_03_single_sar_vqa(self, mock_detect):
        """TEST 3: One SAR image, 'Analyze this SAR image.' -> sar_vqa"""
        mock_detect.return_value = {"modality": "sar", "confidence": 0.95, "evidence": ["mock sar"]}
        res = run_satquery([SAR_FILE], "Analyze this SAR image.", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "sar_vqa")
        self.assertEqual(res["modalities"][0]["modality"], "sar")

    @patch("router.detect_modality")
    @patch("router.subprocess.run")
    def test_04_two_optical_change_vqa(self, mock_subproc, mock_detect):
        """TEST 4: Two optical images, 'What changed between these images?' -> change_vqa with optical venv & GeoChat"""
        mock_detect.side_effect = [
            {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical 1"]},
            {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical 2"]},
        ]
        mock_subproc.return_value.returncode = 0
        mock_subproc.return_value.stdout = json.dumps({
            "success": True,
            "modality": "optical",
            "backend": "GeoChat-7B",
            "executable": OPTICAL_PYTHON,
            "combined_change_interpretation": "Mock optical change interpretation",
            "pixel_difference_result": "Mock optical pixel diff",
            "change_map_png": r"D:\satQai\outputs\optical_change_mock.png",
            "raw_diff_npy": r"D:\satQai\outputs\optical_diff_mock.npy",
            "diff_stats": {"significant_change_pct": 5.2},
            "before_caption": "Mock optical pre description",
            "after_caption": "Mock optical post description",
        })
        mock_subproc.return_value.stderr = ""

        res = run_satquery([OPTICAL_FILE, OPTICAL_FILE2], "What changed between these images?", mock_specialist=False)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "change_vqa")
        self.assertEqual(res["modalities"][0]["modality"], "optical")
        self.assertEqual(res["modalities"][1]["modality"], "optical")

        called_cmd = mock_subproc.call_args[0][0]
        self.assertEqual(os.path.normcase(called_cmd[0]), os.path.normcase(OPTICAL_PYTHON))
        self.assertIn("GeoChat-7B", res["specialists_used"])
        self.assertNotIn("Qwen2-VL-2B", res["specialists_used"])

    @patch("router.detect_modality")
    def test_05_two_optical_compare_before_after(self, mock_detect):
        """TEST 5: Two optical images, 'Compare the before and after scenes.' -> change_vqa"""
        mock_detect.return_value = {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical"]}
        res = run_satquery([OPTICAL_FILE, OPTICAL_FILE2], "Compare the before and after scenes.", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "change_vqa")

    @patch("router.detect_modality")
    def test_06_optical_sar_fusion(self, mock_detect):
        """TEST 6: Optical + SAR, 'Use both images to identify built-up and water.' -> optical_sar_fusion"""
        mock_detect.side_effect = [
            {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical"]},
            {"modality": "sar", "confidence": 0.95, "evidence": ["mock sar"]},
        ]
        res = run_satquery([OPTICAL_FILE, SAR_FILE], "Use both images to identify built-up and water.", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "optical_sar_fusion")

    @patch("router.detect_modality")
    def test_07_optical_sar_ambiguous_compare(self, mock_detect):
        """TEST 7: Optical + SAR, 'Compare these images.' -> clarification_required"""
        mock_detect.side_effect = [
            {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical"]},
            {"modality": "sar", "confidence": 0.95, "evidence": ["mock sar"]},
        ]
        res = run_satquery([OPTICAL_FILE, SAR_FILE], "Compare these images.", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "clarification_required")
        self.assertIn("cross-modal fusion", res["answer"])

    @patch("router.detect_modality")
    def test_08_unknown_modality(self, mock_detect):
        """TEST 8: Unknown modality, 'What is in this image?' -> clarification_required"""
        mock_detect.return_value = {"modality": "unknown", "confidence": 0.0, "evidence": ["inconclusive"]}
        res = run_satquery([OPTICAL_FILE], "What is in this image?", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "clarification_required")
        self.assertEqual(res["modalities"][0]["modality"], "unknown")
        self.assertIn("could not definitively determine sensor type", res["answer"])

    def test_09_missing_image_path(self):
        """TEST 9: Missing image path -> structured error with uniform schema"""
        bad_path = r"D:\satQai\non_existent_image_12345.tif"
        res = run_satquery([bad_path], "Describe this image.")
        self.assert_standard_schema(res, expect_error=True)
        self.assertEqual(res["task"], "error")
        self.assertEqual(res["error"]["type"], "FileNotFoundError")
        self.assertIn("does not exist", res["error"]["message"])

    def test_10_three_images_unsupported(self):
        """TEST 10: Three images -> clarification_required / unsupported"""
        res = run_satquery([OPTICAL_FILE, OPTICAL_FILE2, SAR_FILE], "Analyze these three images.", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "clarification_required")
        self.assertIn("currently supports single-image analysis", res["answer"])

    @patch("router.detect_modality")
    def test_11_single_optical_caption_alternate_phrasing(self, mock_detect):
        """TEST 11: One optical image, 'Give me a caption of the land cover.' -> optical_caption"""
        mock_detect.return_value = {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical"]}
        res = run_satquery([OPTICAL_FILE], "Give me a caption of the land cover.", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "optical_caption")

    @patch("router.detect_modality")
    @patch("router.subprocess.run")
    def test_12_two_sar_change_vqa(self, mock_subproc, mock_detect):
        """TEST 12: Two SAR images, 'What changed between these SAR images?' -> change_vqa with SAR venv & Qwen2-VL"""
        mock_detect.side_effect = [
            {"modality": "sar", "confidence": 0.95, "evidence": ["mock sar 1"]},
            {"modality": "sar", "confidence": 0.95, "evidence": ["mock sar 2"]},
        ]
        mock_subproc.return_value.returncode = 0
        mock_subproc.return_value.stdout = json.dumps({
            "success": True,
            "modality": "sar",
            "backend": "Qwen2-VL-2B",
            "executable": SAR_PYTHON,
            "combined_change_interpretation": "Mock SAR change interpretation",
            "pixel_difference_result": "Mock SAR pixel diff",
            "change_map_png": r"D:\satQai\outputs\sar_change_mock.png",
            "raw_diff_npy": r"D:\satQai\outputs\sar_diff_mock.npy",
            "diff_stats": {"significant_change_pct": 21.9},
            "before_caption": "Mock SAR pre description",
            "after_caption": "Mock SAR post description",
        })
        mock_subproc.return_value.stderr = ""

        res = run_satquery([SAR_FILE, SAR_FILE2], "What changed between these SAR images?", mock_specialist=False)
        self.assert_standard_schema(res)

        # 1. Assert task
        self.assertEqual(res["task"], "change_vqa")

        # 2. Capture actual subprocess command constructed by router
        self.assertTrue(mock_subproc.called, "subprocess.run was not invoked by execute_specialist")
        called_cmd = mock_subproc.call_args[0][0]
        called_executable = os.path.normcase(called_cmd[0])
        expected_sar_exec = os.path.normcase(SAR_PYTHON)
        optical_exec = os.path.normcase(OPTICAL_PYTHON)

        # 3. Assert SAR executable is used and optical executable is NOT used
        self.assertEqual(
            called_executable,
            expected_sar_exec,
            f"Expected SAR executable {expected_sar_exec}, but got {called_executable}"
        )
        self.assertNotEqual(
            called_executable,
            optical_exec,
            "Test FAILED: Optical executable was improperly selected for SAR-SAR change query!"
        )

        # 4. Assert command line flags passed to change_vqa_cli
        self.assertIn("--modality", called_cmd)
        mod_idx = called_cmd.index("--modality")
        self.assertEqual(called_cmd[mod_idx + 1], "sar")

        # 5. Assert specialist / backend is Qwen2-VL, NOT GeoChat
        self.assertIn("Qwen2-VL-2B", res["specialists_used"])
        self.assertNotIn("GeoChat-7B", res["specialists_used"])
        self.assertEqual(res["metadata"].get("backend"), "Qwen2-VL-2B")
        self.assertEqual(os.path.normcase(res["metadata"].get("executable")), expected_sar_exec)

    @patch("router.detect_modality")
    @patch("router.requests.post")
    def test_13_simulated_specialist_failure(self, mock_post, mock_detect):
        """TEST 13: Simulated specialist server failure -> structured ServerUnavailableError with normalized schema"""
        mock_detect.return_value = {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical"]}
        mock_post.side_effect = Exception("CUDA out of memory in specialist engine")

        res = run_satquery([OPTICAL_FILE], "What buildings are visible?", mock_specialist=False)

        self.assert_standard_schema(res, expect_error=True)
        self.assertIn(res["task"], ["error", "optical_vqa"])
        self.assertIn(res["error"]["type"], ["ServerUnavailableError", "SpecialistExecutionError"])
        self.assertIn("CUDA out of memory", res["error"]["message"])
        self.assertNotIn("Traceback", res["answer"])
        self.assertIn("failed during execution", res["answer"])
        self.assertIn("router_total_time_s", res["timing"])
        self.assertIn("specialist_execution_time_s", res["timing"])

    @patch("router.detect_modality")
    def test_14_optical_sar_change_clarification(self, mock_detect):
        """TEST 14: Optical + SAR with change query -> clarification_required with normalized schema"""
        mock_detect.side_effect = [
            {"modality": "optical", "confidence": 0.95, "evidence": ["mock optical"]},
            {"modality": "sar", "confidence": 0.95, "evidence": ["mock sar"]},
        ]
        res = run_satquery([OPTICAL_FILE, SAR_FILE], "What changed between these images?", mock_specialist=True)
        self.assert_standard_schema(res)
        self.assertEqual(res["task"], "clarification_required")
        self.assertIn("Temporal change comparison requires two images of the same modality", res["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
