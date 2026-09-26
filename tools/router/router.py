#!/usr/bin/env python3
"""SatQueryAI Agentic Controller & Router.

Accepts user query and one or more image paths, executes deterministic modality detection,
classifies the analytical task according to strict routing rules, orchestrates specialist
tools via isolated subprocesses, logs all actions, and returns standardized JSON.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
import warnings
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*geotransform.*")
warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")

class ServerUnavailableError(Exception):
    """Raised when a persistent model server is unreachable."""
    pass

OPTICAL_SERVER_URL = "http://127.0.0.1:8001"
SAR_SERVER_URL = "http://127.0.0.1:8002" 

# System paths
# Enforce offline mode for Hugging Face Hub across all routed subprocesses
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = r"D:\satQai"
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
ROUTER_LOG_FILE = os.path.join(LOGS_DIR, "router.log")

OPTICAL_PYTHON = r"D:\satQai\tools\optical_vqa\venv\Scripts\python.exe"
SAR_PYTHON = r"D:\satQai\tools\sar_vqa\venv\Scripts\python.exe"
MAIN_PYTHON = r"D:\satQai\venv\Scripts\python.exe"

OPTICAL_CLI = r"D:\satQai\tools\optical_vqa\optical_vqa_cli.py"
SAR_CLI = r"D:\satQai\tools\sar_vqa\sar_vqa_cli.py"
CHANGE_CLI = r"D:\satQai\tools\change_vqa\change_vqa_cli.py"
FUSION_SCRIPT = r"D:\satQai\tools\fusion\fusion.py"
GIS_PREPROCESS_DIR = r"D:\satQai\tools\gis_preprocess"

# Ensure GIS preprocess is accessible for detect_modality
if GIS_PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, GIS_PREPROCESS_DIR)

try:
    from detect_modality import detect_modality
except ImportError:
    detect_modality = None


# Configure router file logger
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

logger = logging.getLogger("satquery_router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(ROUTER_LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s UTC] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def _has_keyword(text: str, patterns: List[str]) -> bool:
    """Check if any regex pattern matches in the query string."""
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False


# Intent keyword regex patterns
CAPTION_PATTERNS = [
    r"\bdescribe\b",
    r"\bcaption\b",
    r"\boverview\b",
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\bwhat does this (image|scene) show\b",
    r"\bwhat is (in|visible in) this image\b",
    r"\btell me about this (image|scene)\b",
]

CHANGE_PATTERNS = [
    r"\bchange\b",
    r"\bchanges\b",
    r"\bdifference\b",
    r"\bdifferences\b",
    r"\bbefore\b.*\bafter\b",
    r"\bafter\b.*\bbefore\b",
    r"\btemporal\b",
    r"\bwhat changed\b",
    r"\bdetect.*change\b",
    r"\bcompare.*time\b",
    r"\bcompare.*scenes\b",
    r"\bover time\b",
    r"\bwhat are the changes\b",
    r"\bchanges here\b",
    r"\bchanges in\b",
]

FUSION_PATTERNS = [
    r"\bboth\b",
    r"\bcombine\b",
    r"\btogether\b",
    r"\bfusion\b",
    r"\bfuse\b",
    r"\bcross-modal\b",
    r"\boptical and sar\b",
    r"\bsar and optical\b",
    r"\buse.*together\b",
    r"\busing both\b",
]


def classify_task(
    image_paths: List[str],
    query: str,
    modalities_info: List[Dict[str, Any]],
) -> Tuple[str, Optional[str]]:
    """Determine the SatQueryAI task category and clarification message based on routing rules.

    Returns:
        (task_category, clarification_message)
    """
    n_images = len(image_paths)

    # Rule: > 2 images -> clarification_required
    if n_images > 2:
        msg = (
            f"SatQueryAI currently supports single-image analysis (optical or SAR) or "
            f"two-image analysis (bi-temporal change or optical-SAR fusion). Received {n_images} images. "
            f"Please provide either 1 image or a pair of 2 images."
        )
        return "clarification_required", msg

    modalities = [m["modality"] for m in modalities_info]

    # CASE 1 & 2: Single Image
    if n_images == 1:
        mod = modalities[0]
        if mod == "unknown":
            msg = (
                f"Modality detector could not definitively determine sensor type for image: "
                f"{image_paths[0]}. Please specify whether the image is optical (visible/multispectral) "
                f"or SAR (radar), or provide metadata."
            )
            return "clarification_required", msg
        elif mod == "optical":
            if _has_keyword(query, CAPTION_PATTERNS):
                return "optical_caption", None
            return "optical_vqa", None
        elif mod == "sar":
            return "sar_vqa", None

    # CASE 3, 4, 5 & Non-georeferenced fallback: Two Images
    if n_images == 2:
        m1, m2 = modalities[0], modalities[1]

        # Case 4, 6, 7: Optical + SAR
        if (m1 == "optical" and m2 == "sar") or (m1 == "sar" and m2 == "optical"):
            # Check explicit fusion intent first
            if _has_keyword(query, FUSION_PATTERNS):
                return "optical_sar_fusion", None

            # Check change intent on heterogeneous pair
            if _has_keyword(query, CHANGE_PATTERNS):
                msg = (
                    "Temporal change comparison requires two images of the same modality; "
                    "an optical + SAR pair should use cross-modal fusion, not change_vqa. "
                    "Did you mean to perform cross-modal fusion (e.g. 'Use optical and SAR together to identify built-up and water')?"
                )
                return "clarification_required", msg

            # Ambiguous query (e.g. "Compare these")
            msg = (
                "An optical image and a SAR image were provided. Do you want cross-modal fusion "
                "of the optical and SAR images (e.g. 'Use both images together to identify land cover'), "
                "or did you intend a temporal/change comparison with a matched-modality pair?"
            )
            return "clarification_required", msg

        # Change Intent for matched modality or non-georeferenced/unknown pair
        if _has_keyword(query, CHANGE_PATTERNS):
            return "change_vqa", None

        # Two optical images without change intent
        if m1 == "optical" and m2 == "optical":
            msg = (
                "Two optical images were provided, but the query does not ask for temporal change comparison. "
                "Please clarify if you want bi-temporal change detection (e.g. 'What changed between these images?')."
            )
            return "clarification_required", msg

        # Two SAR images without change intent
        if m1 == "sar" and m2 == "sar":
            msg = (
                "Two SAR images were provided, but the query does not ask for temporal change comparison. "
                "Please clarify if you want radar change detection (e.g. 'What changed between these SAR images?')."
            )
            return "clarification_required", msg

        # Non-georeferenced / unknown pairs without change intent
        msg = (
            "Two images were provided, but the query does not specify whether you want temporal change detection "
            "(e.g. 'What changed between these images?') or cross-modal fusion."
        )
        return "clarification_required", msg

    return "clarification_required", "Could not determine appropriate specialist for the provided inputs."


def execute_specialist(
    task: str,
    image_paths: List[str],
    query: str,
    modalities_info: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute the selected specialist via isolated subprocess with timeout and logging."""
    t0 = time.time()
    specialist_name = task
    subproc_rc = 0
    err_info = None

    result: Dict[str, Any] = {
        "answer": "",
        "evidence": [],
        "artifacts": [],
        "metadata": {},
        "specialists_used": [],
    }

    try:
        if task == "optical_caption":
            specialist_name = "GeoChat-7B (optical_caption)"
            url = f"{OPTICAL_SERVER_URL}/caption"
            payload = {"image_path": image_paths[0], "prompt": query, "max_new_tokens": 256}
            try:
                resp = requests.post(url, json=payload, timeout=120)
                if resp.status_code != 200:
                    raise RuntimeError(f"Optical server error (HTTP {resp.status_code}): {resp.text}")
                out_json = resp.json()
            except Exception as exc:
                raise ServerUnavailableError(f"Optical specialist server unavailable at {url}: {exc}") from exc

            result["answer"] = out_json.get("caption", "")
            result["specialists_used"] = ["GeoChat-7B"]
            result["metadata"] = {
                "load_time_s": out_json.get("load_time_s", 0.0),
                "inference_time_s": out_json.get("inference_time_s", 0.0),
                "server_url": OPTICAL_SERVER_URL,
            }

        elif task == "optical_vqa":
            specialist_name = "GeoChat-7B (optical_vqa)"
            url = f"{OPTICAL_SERVER_URL}/vqa"
            payload = {"image_path": image_paths[0], "question": query, "max_new_tokens": 256}
            try:
                resp = requests.post(url, json=payload, timeout=120)
                if resp.status_code != 200:
                    raise RuntimeError(f"Optical server error (HTTP {resp.status_code}): {resp.text}")
                out_json = resp.json()
            except Exception as exc:
                raise ServerUnavailableError(f"Optical specialist server unavailable at {url}: {exc}") from exc

            result["answer"] = out_json.get("answer", "")
            result["evidence"] = [
                f"confidence: {out_json.get('confidence')}",
                f"explanation: {out_json.get('explanation')}",
            ]
            result["specialists_used"] = ["GeoChat-7B"]
            result["metadata"] = {
                "load_time_s": out_json.get("load_time_s", 0.0),
                "inference_time_s": out_json.get("inference_time_s", 0.0),
                "server_url": OPTICAL_SERVER_URL,
            }

        elif task == "sar_vqa":
            specialist_name = "Qwen2-VL-2B (sar_vqa | subprocess)"
            out_json = os.path.join(OUTPUTS_DIR, f"sar_vqa_turn_{int(time.time())}.json")
            cmd = [
                SAR_PYTHON,
                SAR_CLI,
                "--image", image_paths[0],
                "--question", query,
                "--max_new_tokens", "256",
                "--output_json", out_json,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            subproc_rc = proc.returncode
            if os.path.exists(out_json):
                with open(out_json, "r", encoding="utf-8") as f:
                    cli_out = json.load(f)
                result["answer"] = cli_out.get("answer", "")
                result["metadata"] = {
                    "load_time_s": cli_out.get("load_time_s", 0.0),
                    "inference_time_s": cli_out.get("inference_time_s", 0.0),
                    "routing_path": "subprocess",
                }
            elif proc.stdout:
                parsed = _extract_json(proc.stdout)
                result["answer"] = parsed.get("answer", proc.stdout.strip())
                result["metadata"] = parsed
                result["metadata"]["routing_path"] = "subprocess"
            else:
                result["answer"] = proc.stderr or "SAR specialist execution failed."
                result["error"] = {"type": "SpecialistExecutionError", "message": proc.stderr}
            result["specialists_used"] = ["Qwen2-VL-2B (subprocess)"]

        elif task == "change_vqa":
            mod = modalities_info[0]["modality"]
            if mod not in ["optical", "sar"]:
                mod = "optical"
            python_exec = OPTICAL_PYTHON if mod == "optical" else SAR_PYTHON
            backend_used = "GeoChat-7B" if mod == "optical" else "Qwen2-VL-2B"
            specialist_name = f"Change-VQA ({backend_used} | {python_exec})"
            cmd = [
                python_exec,
                CHANGE_CLI,
                "--pre", image_paths[0],
                "--post", image_paths[1],
                "--modality", mod,
                "--query", query,
                "--out_dir", OUTPUTS_DIR,
            ]
            logger.info(f"Executing Change-VQA specialist: exec={python_exec} | backend={backend_used} | cmd={cmd}")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            subproc_rc = proc.returncode
            if proc.returncode != 0:
                clean_err = re.sub(r'[\w\\/]+[\\/]rasterio[\\/].*?NotGeoreferencedWarning.*?\n?', '', proc.stderr)
                raise RuntimeError(f"Change VQA specialist failed (rc={proc.returncode}): {clean_err.strip()}")
            out_json = _extract_json(proc.stdout)
            result["answer"] = out_json.get("combined_change_interpretation", proc.stdout.strip())
            evidence_items = []
            if out_json.get("mode"):
                evidence_items.append(f"Pipeline Mode: {out_json.get('mode')}")
            if out_json.get("pixel_difference_result"):
                evidence_items.append(out_json.get("pixel_difference_result"))
            if out_json.get("before_caption"):
                evidence_items.append(f"Pre-event description ({backend_used}): {out_json.get('before_caption')}")
            if out_json.get("after_caption"):
                evidence_items.append(f"Post-event description ({backend_used}): {out_json.get('after_caption')}")
            result["evidence"] = evidence_items
            result["artifacts"] = [out_json.get("change_map_png", ""), out_json.get("raw_diff_npy", "")]
            result["specialists_used"] = ["Change-VQA", backend_used]
            result["metadata"] = {
                "mode": out_json.get("mode", "Geospatial Change-VQA"),
                "modality": mod,
                "backend": backend_used,
                "executable": python_exec,
                "diff_stats": out_json.get("diff_stats", {}),
                "registration_info": out_json.get("registration_info", {}),
                "before_caption": out_json.get("before_caption", ""),
                "after_caption": out_json.get("after_caption", ""),
            }

        elif task == "optical_sar_fusion":
            specialist_name = "Optical-SAR Fusion (fusion.py)"
            # Find optical vs sar paths
            if modalities_info[0]["modality"] == "optical":
                opt_path, sar_path = image_paths[0], image_paths[1]
            else:
                opt_path, sar_path = image_paths[1], image_paths[0]

            cmd = [
                MAIN_PYTHON,
                FUSION_SCRIPT,
                "--optical", opt_path,
                "--sar", sar_path,
                "--query", query,
                "--output_dir", OUTPUTS_DIR,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            subproc_rc = proc.returncode
            if proc.returncode != 0:
                raise RuntimeError(f"Fusion specialist failed (rc={proc.returncode}): {proc.stderr}")
            
            # Read generated fusion_report.json
            report_path = os.path.join(OUTPUTS_DIR, "fusion_report.json")
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                synth = report.get("cross_modal_synthesis", {})
                result["answer"] = synth.get("synthesis_narrative", "")
                result["evidence"] = [
                    f"Confirmed Built-up: {report.get('quantitative_metrics', {}).get('cross_modal_agreement_relative_to_valid_overlap', {}).get('confirmed_built_up_pct')}%",
                    f"Confirmed Water: {report.get('quantitative_metrics', {}).get('cross_modal_agreement_relative_to_valid_overlap', {}).get('confirmed_water_pct')}%",
                ]
                result["artifacts"] = [
                    report.get("inputs", {}).get("fusion_map_png", ""),
                    report.get("inputs", {}).get("fusion_mask_tif", ""),
                    report_path,
                ]
                metrics = report.get("quantitative_metrics", {})
                result["metadata"] = {
                    "load_time_s": {
                        "optical_vqa": report.get("optical_vqa", {}).get("load_time_s"),
                        "sar_vqa": report.get("sar_vqa", {}).get("load_time_s"),
                    },
                    "inference_time_s": {
                        "optical_vqa": report.get("optical_vqa", {}).get("inference_time_s"),
                        "sar_vqa": report.get("sar_vqa", {}).get("inference_time_s"),
                    },
                    **metrics,
                }
            else:
                result["answer"] = proc.stdout.strip()
            result["specialists_used"] = ["Optical-SAR Fusion", "GeoChat-7B", "Qwen2-VL-2B"]

    except Exception as exc:
        err_info = str(exc)
        err_type = "ServerUnavailableError" if isinstance(exc, ServerUnavailableError) or "unavailable" in err_info.lower() else "SpecialistExecutionError"
        result["answer"] = f"Specialist '{specialist_name}' failed during execution: {err_info}"
        result["error"] = {
            "type": err_type,
            "message": err_info,
        }
        result["specialists_used"] = [specialist_name]

    dur = time.time() - t0
    # Audit log
    logger.info(
        f"Executed specialist: '{specialist_name}' | Task: {task} | Duration: {dur:.2f}s | "
        f"ReturnCode: {subproc_rc} | Error: {err_info or 'None'}"
    )
    result["timing"] = {"specialist_execution_time_s": round(dur, 2)}
    return result


def _extract_json(stdout: str) -> Dict[str, Any]:
    """Extract JSON block from subprocess stdout."""
    stdout = stdout.strip()
    json_start = stdout.find("{")
    json_end = stdout.rfind("}")
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(stdout[json_start : json_end + 1])
        except Exception:
            pass
    return {}


def run_satquery(image_paths: List[str], query: str, mock_specialist: bool = False) -> Dict[str, Any]:
    """Controller public API: process input images and natural-language query.

    Args:
        image_paths: List of absolute or relative file paths to input satellite images.
        query: User's natural language question or instruction.
        mock_specialist: If True, skips actual VLM subprocess invocation (for fast unit tests).

    Returns:
        Standardized JSON-serializable dictionary.
    """
    t_start = time.time()

    # 1. Input Validation
    if not image_paths or len(image_paths) == 0:
        t_total = round(time.time() - t_start, 2)
        err_res = {
            "task": "error",
            "modalities": [],
            "specialists_used": [],
            "answer": "Error: At least one image path must be provided.",
            "evidence": [],
            "artifacts": [],
            "error": {"type": "ValueError", "message": "At least one image path must be provided."},
            "metadata": {"query": query},
            "timing": {"router_total_time_s": t_total},
        }
        logger.error(f"Input validation failure: Empty image_paths list for query '{query}'")
        return err_res

    missing_paths = [p for p in image_paths if not os.path.exists(p)]
    if missing_paths:
        t_total = round(time.time() - t_start, 2)
        err_res = {
            "task": "error",
            "modalities": [],
            "specialists_used": [],
            "answer": f"Error: Image path does not exist: {', '.join(missing_paths)}",
            "evidence": [],
            "artifacts": [],
            "error": {
                "type": "FileNotFoundError",
                "message": f"Image path does not exist: {', '.join(missing_paths)}",
            },
            "metadata": {"image_paths": image_paths, "query": query},
            "timing": {"router_total_time_s": t_total},
        }
        logger.error(f"Input validation failure: Missing files: {missing_paths}")
        return err_res

    # 2. Modality Detection
    modalities_info: List[Dict[str, Any]] = []
    for p in image_paths:
        abs_p = os.path.abspath(p)
        if detect_modality:
            det = detect_modality(abs_p)
            modalities_info.append({
                "path": abs_p,
                "modality": det.get("modality", "unknown"),
                "confidence": det.get("confidence", 0.0),
                "evidence": det.get("evidence", []),
            })
        else:
            modalities_info.append({
                "path": abs_p,
                "modality": "unknown",
                "confidence": 0.0,
                "evidence": ["detect_modality module not available"],
            })

    # 3. Task Decision Engine
    task, clar_msg = classify_task(image_paths, query, modalities_info)

    logger.info(
        f"Routing Decision: Inputs={image_paths} | Modalities={[m['modality'] for m in modalities_info]} | "
        f"Query='{query}' -> Task={task}"
    )

    # 4. Handle Clarification Required
    if task == "clarification_required":
        t_total = round(time.time() - t_start, 2)
        return {
            "task": "clarification_required",
            "modalities": modalities_info,
            "specialists_used": [],
            "answer": clar_msg or "Please clarify your request.",
            "evidence": [],
            "artifacts": [],
            "metadata": {"query": query, "image_count": len(image_paths)},
            "timing": {
                "router_total_time_s": t_total,
                "router_decision_time_s": t_total,
            },
        }

    # 5. Execute Specialist
    if mock_specialist:
        t_total = round(time.time() - t_start, 2)
        return {
            "task": task,
            "modalities": modalities_info,
            "specialists_used": [f"Mock_{task}"],
            "answer": f"Mock response for task: {task}",
            "evidence": [],
            "artifacts": [],
            "metadata": {"query": query, "mock": True},
            "timing": {
                "router_total_time_s": t_total,
                "router_decision_time_s": t_total,
            },
        }

    spec_res = execute_specialist(task, image_paths, query, modalities_info)
    t_total = round(time.time() - t_start, 2)

    res_dict = {
        "task": "error" if "error" in spec_res else task,
        "modalities": modalities_info,
        "specialists_used": spec_res.get("specialists_used", []),
        "answer": spec_res.get("answer", ""),
        "evidence": spec_res.get("evidence", []),
        "artifacts": spec_res.get("artifacts", []),
        "metadata": spec_res.get("metadata", {}),
        "timing": {
            "router_total_time_s": t_total,
            **spec_res.get("timing", {}),
        },
    }
    if "error" in spec_res:
        res_dict["error"] = spec_res["error"]

    return res_dict


# Backward-compatible alias
route_query = run_satquery
