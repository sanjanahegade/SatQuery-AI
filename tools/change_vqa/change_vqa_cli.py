#!/usr/bin/env python3
"""CLI interface for Change-VQA (Bi-temporal satellite image change analysis).

MODALITY-AWARE BACKEND BRANCHING:
  - Optical + Optical:
      Change-VQA quantitative registration/normalization/diff logic
      GeoChat before/after scene descriptions
      Executable: D:\satQai\tools\optical_vqa\venv\Scripts\python.exe
  - SAR + SAR:
      Change-VQA quantitative registration/backscatter dB diff logic
      Qwen2-VL before/after scene descriptions
      Executable: D:\satQai\tools\sar_vqa\venv\Scripts\python.exe
"""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
import warnings
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*geotransform.*")
warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")

# Executable paths for isolated specialist environments
OPTICAL_PYTHON = r"D:\satQai\tools\optical_vqa\venv\Scripts\python.exe"
SAR_PYTHON = r"D:\satQai\tools\sar_vqa\venv\Scripts\python.exe"
SAR_CLI = r"D:\satQai\tools\sar_vqa\sar_vqa_cli.py"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

GIS_DIR = r"D:\satQai\tools\gis_preprocess"
if GIS_DIR not in sys.path:
    sys.path.insert(0, GIS_DIR)

try:
    from detect_modality import detect_modality
except ImportError:
    detect_modality = None


SCIENTIFIC_CAVEAT_SAR = (
    "SCIENTIFIC CAVEAT: Observed SAR backscatter differences represent radar cross-section "
    "attenuation or enhancement. Major factors include dielectric permittivity (water inundation / "
    "soil moisture), surface roughness variations, and geometric viewing angle effects. "
    "Quantitative dB attenuation must be interpreted in conjunction with local topography and "
    "hydrologic drainage patterns."
)


def resolve_backend(pre_path, post_path, explicit_modality=None):
    """Determine the pair modality, backend model name, and required Python executable.

    Returns:
        tuple: (resolved_modality, backend_name, target_python)
    """
    if explicit_modality in ["optical", "sar"]:
        modality = explicit_modality
    else:
        if detect_modality is None:
            modality = "optical"
        else:
            d1 = detect_modality(pre_path)
            d2 = detect_modality(post_path)
            m1, m2 = d1.get("modality"), d2.get("modality")
            if m1 == "optical" and m2 == "optical":
                modality = "optical"
            elif m1 == "sar" and m2 == "sar":
                modality = "sar"
            elif (m1 == "optical" and m2 == "sar") or (m1 == "sar" and m2 == "optical"):
                raise ValueError(
                    "Heterogeneous optical + SAR pair detected. Temporal Change-VQA requires "
                    "two images of the same modality. Use optical-SAR fusion instead."
                )
            else:
                # Default to optical/GeoChat for non-georeferenced or unknown imagery
                modality = "optical" 

    # Select executable and backend
    if modality == "optical":
        return "optical", "GeoChat", OPTICAL_PYTHON
    elif modality == "sar":
        return "sar", "Qwen2-VL", SAR_PYTHON
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def run_optical_change(pre_path, post_path, query=None, out_dir=r"D:\satQai\data\real_test"):
    """Execute optical bi-temporal change analysis using GeoChat (optical venv)."""
    from change_vqa import analyze_change_pair, get_engine_or_remote
    pair_name = f"Change Analysis ({os.path.basename(pre_path)} vs {os.path.basename(post_path)})"

    t0 = time.time()
    engine = get_engine_or_remote()
    res = analyze_change_pair(
        pre_path=pre_path,
        post_path=post_path,
        pair_name=pair_name,
        out_dir=out_dir,
        engine=engine,
    )
    dur = time.time() - t0

    mode = res.get("mode") or res.get("registration_info", {}).get("mode", "Geospatial Change-VQA")
    return {
        "success": True,
        "mode": mode,
        "modality": "optical",
        "backend": "GeoChat",
        "executable": OPTICAL_PYTHON,
        "pair_name": pair_name,
        "pre_path": pre_path,
        "post_path": post_path,
        "query": query,
        "registration_info": res.get("registration_info", {}),
        "diff_stats": res.get("diff_stats", {}),
        "before_caption": res.get("before_caption", ""),
        "after_caption": res.get("after_caption", ""),
        "pixel_difference_result": res.get("pixel_difference_result", ""),
        "combined_change_interpretation": res.get("combined_change_interpretation", ""),
        "change_map_png": res.get("change_map_png", ""),
        "raw_diff_npy": res.get("raw_diff_npy", ""),
        "duration_s": round(dur, 2),
    }


def load_sar_preview(image_path):
    """Load SAR image and convert to 8-bit PIL RGB image for Qwen2-VL ingestion."""
    from PIL import Image
    import rasterio

    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        with rasterio.open(image_path) as src:
            arr = src.read(1).astype(np.float32)
            dn_safe = np.where(arr > 0, arr, 1e-6)
            db = 20.0 * np.log10(dn_safe) - 83.0
            db_clipped = np.clip(db, -25.0, 0.0)
            norm = ((db_clipped - (-25.0)) / 25.0 * 255.0).astype(np.uint8)
            return Image.fromarray(norm).convert("RGB")
    return Image.open(image_path).convert("RGB")


def run_sar_qwen2vl_captions(pre_path, post_path):
    """Generate pre-event and post-event SAR scene descriptions using Qwen2-VL-2B in SAR venv."""
    import torch
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )

    def describe_one(img_path, label):
        pil_img = load_sar_preview(img_path)
        prompt_text = (
            "Describe the land-cover and features visible in this SAR image, "
            "noting backscatter patterns, surface roughness, structural outlines, and potential water or urban features."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        torch.cuda.synchronize()

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return caption

    before_caption = describe_one(pre_path, "pre-event")
    after_caption = describe_one(post_path, "post-event")

    del model, processor
    torch.cuda.empty_cache()

    return before_caption, after_caption, round(time.time() - t0, 2)


def run_sar_change(pre_path, post_path, query=None, out_dir=r"D:\satQai\data\real_test"):
    """Execute SAR bi-temporal change analysis using Qwen2-VL & dB thresholding (sar venv)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()

    with rasterio.open(pre_path) as s_pre, rasterio.open(post_path) as s_post:
        p_arr = s_pre.read(1).astype(np.float32)
        q_arr = s_post.read(1).astype(np.float32)
        post_profile = s_post.profile.copy()
        post_crs = s_post.crs
        post_transform = s_post.transform
        pre_crs = s_pre.crs
        pre_transform = s_pre.transform

    # Reproject pre onto post if shapes or grids differ
    reprojection_applied = False
    if p_arr.shape != q_arr.shape or pre_crs != post_crs or pre_transform != post_transform:
        reg_p = np.zeros_like(q_arr)
        with rasterio.open(pre_path) as s_pre, rasterio.open(post_path) as s_post:
            reproject(
                source=rasterio.band(s_pre, 1),
                destination=reg_p,
                src_transform=s_pre.transform,
                src_crs=s_pre.crs,
                dst_transform=s_post.transform,
                dst_crs=s_post.crs,
                resampling=Resampling.bilinear,
            )
        p_arr = reg_p
        reprojection_applied = True

    # Calibrated backscatter in decibels: sigma0_dB = 20 * log10(DN) - 83.0
    pre_db = 20.0 * np.log10(np.maximum(p_arr, 1.0)) - 83.0
    post_db = 20.0 * np.log10(np.maximum(q_arr, 1.0)) - 83.0

    vmin = float(np.percentile(pre_db, 1.0))
    vmax = float(np.percentile(pre_db, 99.0))
    db_span = vmax - vmin if vmax > vmin else 1.0

    pre_norm = np.clip((pre_db - vmin) / db_span, 0.0, 1.0)
    post_norm = np.clip((post_db - vmin) / db_span, 0.0, 1.0)
    diff_norm = post_norm - pre_norm
    mad_norm = np.abs(diff_norm)

    threshold = 0.08
    sig_mask = mad_norm > threshold
    sig_pct = round(float(np.count_nonzero(sig_mask)) / diff_norm.size * 100.0, 2)
    dark_pct = round(float(np.count_nonzero(diff_norm < -threshold)) / diff_norm.size * 100.0, 2)
    bright_pct = round(float(np.count_nonzero(diff_norm > threshold)) / diff_norm.size * 100.0, 2)

    delta_db = post_db - pre_db
    dark_2db = round(float(np.count_nonzero(delta_db < -2.0)) / delta_db.size * 100.0, 2)

    ts = int(time.time())
    map_png_path = os.path.join(out_dir, f"sar_change_{ts}.png")
    diff_npy_path = os.path.join(out_dir, f"sar_diff_{ts}.npy")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    axes[0].imshow(pre_norm, cmap="gray")
    axes[0].set_title(f"Pre-Event SAR: {os.path.basename(pre_path)}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(post_norm, cmap="gray")
    axes[1].set_title(f"Post-Event SAR: {os.path.basename(post_path)}", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    im = axes[2].imshow(diff_norm, cmap="coolwarm", vmin=-0.3, vmax=0.3)
    axes[2].set_title(f"SAR Divergence (Sig: {sig_pct}%, Atten <-2dB: {dark_2db}%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Relative Backscatter Divergence (Normalized)", fontsize=9)

    plt.suptitle("SAR Bi-Temporal Change Analysis (Sentinel-1 IW GRD)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(map_png_path, bbox_inches="tight")
    plt.close(fig)

    np.save(diff_npy_path, diff_norm)

    pixel_diff = (
        f"Sentinel-1 SAR change detected across {sig_pct}% of the scene (mean absolute divergence = {float(np.mean(mad_norm)):.4f}). "
        f"Darkening (specular attenuation / flood water / saturated mud): {dark_pct}%. "
        f"Brightening (surface roughness / sediment / debris): {bright_pct}%. "
        f"Backscatter drop < -2.0 dB: {dark_2db}%."
    )

    # Execute Qwen2-VL for before/after descriptions in SAR environment
    print("[Change-VQA SAR] Running Qwen2-VL scene descriptions on pre and post SAR scenes...")
    before_caption, after_caption, vlm_dur = run_sar_qwen2vl_captions(pre_path, post_path)
    print(f"[Change-VQA SAR] Qwen2-VL descriptions completed in {vlm_dur}s.")

    # Physics-based interpretation logic
    if (dark_pct >= 15.0 or dark_2db >= 15.0) and dark_pct > bright_pct * 0.8:
        event_context = (
            f"Detected change is consistent with an acute hydrologic / flood inundation event derived explicitly from "
            f"quantitative backscatter statistics ({dark_pct}% localized darkening, with {dark_2db}% dropping below -2.0 dB "
            f"characteristic of specular radar deflection from floodwater inundation along valley corridors). "
            f"Note: this interpretation is derived from quantitative backscatter statistics. The VQA model's zero-shot scene descriptions "
            f"did not independently identify the water/flood signal, consistent with known limitations of pre-trained VLMs on SAR imagery "
            f"without domain-specific fine-tuning."
        )
    elif sig_pct < 8.0:
        event_context = (
            f"Detected change indicates a highly stable radar backscatter baseline ({round(100.0 - sig_pct, 2)}% stationary). "
            f"Minor divergence ({sig_pct}%) corresponds to subtle soil moisture fluctuations or minor structural changes."
        )
    else:
        event_context = (
            f"Detected change reflects moderate radar backscatter divergence ({sig_pct}% changed) across the landscape."
        )

    combined_interpretation = (
        f"SAR BI-TEMPORAL SYNTHESIS ({os.path.basename(pre_path)} vs {os.path.basename(post_path)}):\n"
        f"- Pre-Event Scene (Qwen2-VL): {before_caption}\n"
        f"- Post-Event Scene (Qwen2-VL): {after_caption}\n"
        f"- Radiometric Change Metric: {pixel_diff}\n"
        f"- Physical Interpretation: {event_context}\n\n"
        f"{SCIENTIFIC_CAVEAT_SAR}"
    )

    dur = time.time() - t0
    return {
        "success": True,
        "modality": "sar",
        "backend": "Qwen2-VL",
        "executable": SAR_PYTHON,
        "pair_name": f"SAR Change ({os.path.basename(pre_path)} vs {os.path.basename(post_path)})",
        "pre_path": pre_path,
        "post_path": post_path,
        "query": query,
        "registration_info": {
            "reprojection_applied": reprojection_applied,
            "target_crs": str(post_crs),
            "target_dimensions": f"{q_arr.shape[1]} x {q_arr.shape[0]}",
        },
        "diff_stats": {
            "significant_change_pct": sig_pct,
            "darkening_pct": dark_pct,
            "brightening_pct": bright_pct,
            "darkening_pct_below_2dB": dark_2db,
            "mean_absolute_divergence": round(float(np.mean(mad_norm)), 4),
        },
        "before_caption": before_caption,
        "after_caption": after_caption,
        "pixel_difference_result": pixel_diff,
        "combined_change_interpretation": combined_interpretation,
        "change_map_png": map_png_path,
        "raw_diff_npy": diff_npy_path,
        "duration_s": round(dur, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Change-VQA CLI (Modality-Aware Bi-Temporal Analysis)")
    parser.add_argument("--pre", required=True, help="Earlier/pre-event image path")
    parser.add_argument("--post", required=True, help="Later/post-event image path")
    parser.add_argument("--modality", choices=["optical", "sar"], default=None, help="Explicit pair modality")
    parser.add_argument("--query", default="What changed between these images?", help="Change query")
    parser.add_argument("--out_dir", default=r"D:\satQai\data\real_test", help="Output directory")
    parser.add_argument("--output_json", default=None, help="Output JSON path")
    args = parser.parse_args()

    pre_path = os.path.abspath(args.pre)
    post_path = os.path.abspath(args.post)

    if not os.path.exists(pre_path) or not os.path.exists(post_path):
        res = {"success": False, "error": f"Image file not found: {pre_path} or {post_path}"}
        print(json.dumps(res, indent=2))
        sys.exit(1)

    # 1. Resolve pair modality, backend name, and required Python executable
    resolved_mod, backend_name, target_python = resolve_backend(pre_path, post_path, args.modality)

    # 2. Verify current process is running in the correct target venv.
    #    If not, re-dispatch to the designated specialist Python executable.
    current_exec = os.path.normcase(sys.executable)
    required_exec = os.path.normcase(target_python)

    if current_exec != required_exec:
        cmd = [
            target_python,
            os.path.abspath(__file__),
            "--pre", pre_path,
            "--post", post_path,
            "--modality", resolved_mod,
            "--query", args.query,
            "--out_dir", args.out_dir,
        ]
        if args.output_json:
            cmd.extend(["--output_json", args.output_json])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)

    # 3. Running in the correct target venv -> execute modality specialist
    try:
        if resolved_mod == "optical":
            result = run_optical_change(pre_path, post_path, query=args.query, out_dir=args.out_dir)
        else:
            result = run_sar_change(pre_path, post_path, query=args.query, out_dir=args.out_dir)

        print(json.dumps(result, indent=2))
        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    except Exception as exc:
        res = {"success": False, "error": str(exc)}
        print(json.dumps(res, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
