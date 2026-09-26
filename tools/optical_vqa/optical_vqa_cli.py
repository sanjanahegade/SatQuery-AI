#!/usr/bin/env python3
"""CLI interface for Optical VQA and Scene Captioning using GeoChat-7B.

This CLI is executed in the isolated optical VQA environment:
    D:\satQai\tools\optical_vqa\venv\Scripts\python.exe

Usage:
    python optical_vqa_cli.py --image <image_path> --mode caption [--prompt <prompt>]
    python optical_vqa_cli.py --image <image_path> --mode vqa --question <question>
    python optical_vqa_cli.py --image <image_path> --prompt <prompt> --output_json <path>
"""

import argparse
import json
import os
import sys
import time

# Force offline mode for Hugging Face Hub to prevent remote HTTP latency
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 

OPTICAL_PYTHON = r"D:\satQai\tools\optical_vqa\venv\Scripts\python.exe"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

GEOCHAT_DIR = r"D:\satQai\GeoChat"
if GEOCHAT_DIR not in sys.path:
    sys.path.insert(0, GEOCHAT_DIR)

try:
    from optical_vqa import caption_image, answer_question, get_engine
except ImportError as e:
    print(json.dumps({"success": False, "error": f"Failed to import optical_vqa: {e}"}))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Optical VQA / Captioning CLI (GeoChat-7B)")
    parser.add_argument("--image", required=True, help="Path to input optical image (GeoTIFF, JPG, PNG)")
    parser.add_argument("--mode", choices=["caption", "vqa"], default="caption", help="Inference mode")
    parser.add_argument("--prompt", default="Describe the land-cover and major objects visible in this image.", help="Prompt for captioning")
    parser.add_argument("--question", default=None, help="Question for VQA mode")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate (default: 256, options: 128, 256)")
    parser.add_argument("--output_json", default=None, help="Path to write output JSON")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        res = {"success": False, "error": f"Image file not found: {image_path}"}
        print(json.dumps(res, indent=2))
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        sys.exit(1)

    t0 = time.time()
    try:
        engine = get_engine()
        load_time = time.time() - t0

        t1 = time.time()
        if args.mode == "vqa" or args.question:
            q = args.question or args.prompt
            ans = answer_question(image_path, q, engine=engine)
            infer_time = time.time() - t1
            result = {
                "success": True,
                "mode": "vqa",
                "image_path": image_path,
                "question": q,
                "answer": ans.get("answer") if isinstance(ans, dict) else str(ans),
                "confidence": ans.get("confidence") if isinstance(ans, dict) else None,
                "explanation": ans.get("confidence_explanation") if isinstance(ans, dict) else None,
                "load_time_s": round(load_time, 2),
                "inference_time_s": round(infer_time, 2),
            }
        else:
            caption = caption_image(image_path, prompt=args.prompt, engine=engine)
            infer_time = time.time() - t1
            result = {
                "success": True,
                "mode": "caption",
                "image_path": image_path,
                "prompt": args.prompt,
                "caption": caption,
                "load_time_s": round(load_time, 2),
                "inference_time_s": round(infer_time, 2),
            }

        print(json.dumps(result, indent=2))

        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    except Exception as exc:
        res = {
            "success": False,
            "error": str(exc),
            "traceback": repr(exc),
        }
        print(json.dumps(res, indent=2))
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
