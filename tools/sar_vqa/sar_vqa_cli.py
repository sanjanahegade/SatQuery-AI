#!/usr/bin/env python3
"""CLI interface for SAR VQA and Scene Captioning using Qwen2-VL-2B-Instruct.

This CLI is executed in the isolated SAR VQA environment:
    D:\satQai\tools\sar_vqa\venv\Scripts\python.exe

Usage:
    python sar_vqa_cli.py --image <image_path> --prompt <prompt> [--output_json <path>]
    python sar_vqa_cli.py --image <image_path> --question <question>
"""

import argparse
import json
import os
import re
import sys

# Force offline mode for Hugging Face Hub to prevent remote HTTP latency
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
import time
import tempfile
import numpy as np
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SAR_PYTHON = r"D:\satQai\tools\sar_vqa\venv\Scripts\python.exe"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def load_sar_image(image_path):
    """Load SAR image, converting GeoTIFF to 8-bit preview if needed."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        # Try rasterio
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                arr = src.read(1).astype(np.float32)
                # Calibrate to dB: 20 * log10(DN) - 83
                dn_safe = np.where(arr > 0, arr, 1e-6)
                db = 20.0 * np.log10(dn_safe) - 83.0
                db_clipped = np.clip(db, -25.0, 0.0)
                norm = ((db_clipped - (-25.0)) / 25.0 * 255.0).astype(np.uint8)
                return Image.fromarray(norm).convert("RGB")
        except Exception:
            pass
    # Standard PIL load
    return Image.open(image_path).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="SAR VQA / Captioning CLI (Qwen2-VL-2B)")
    parser.add_argument("--image", required=True, help="Path to input SAR image (GeoTIFF, PNG, JPG)")
    parser.add_argument("--prompt", default=None, help="Prompt for SAR captioning")
    parser.add_argument("--question", default=None, help="Natural language question for SAR VQA")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate (default: 256, options: 128, 256)")
    parser.add_argument("--output_json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        res = {"success": False, "error": f"Image not found: {image_path}"}
        print(json.dumps(res, indent=2))
        sys.exit(1)

    prompt_text = args.question or args.prompt or "Describe this SAR image in detail, noting backscatter patterns, surface roughness, structural outlines, and potential water or urban features."

    t0 = time.time()
    try:
        pil_img = load_sar_image(image_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="cuda",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        load_time = time.time() - t0

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

        t1 = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        torch.cuda.synchronize()
        infer_time = time.time() - t1

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # Clean up trailing incomplete sentence if token budget was reached
        cleaned_output = raw_output
        if not re.search(r"[.!?]\s*$", cleaned_output) and len(cleaned_output) > 20:
            last_punc = max(
                cleaned_output.rfind(". "),
                cleaned_output.rfind(".\n"),
                cleaned_output.rfind("! "),
                cleaned_output.rfind("? "),
            )
            if last_punc > 0:
                cleaned_output = cleaned_output[: last_punc + 1].strip()

        result = {
            "success": True,
            "image_path": image_path,
            "prompt": prompt_text,
            "answer": cleaned_output,
            "load_time_s": round(load_time, 2),
            "inference_time_s": round(infer_time, 2),
        }
        print(json.dumps(result, indent=2))

        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    except Exception as exc:
        res = {"success": False, "error": str(exc), "traceback": repr(exc)}
        print(json.dumps(res, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
