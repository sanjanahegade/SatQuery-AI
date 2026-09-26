#!/usr/bin/env python3
"""FastAPI server for SAR VQA and Scene Captioning using Qwen2-VL-2B-Instruct.

Runs in D:\satQai\tools\sar_vqa\venv on port 8002.
Loads Qwen2-VL-2B ONCE into memory at server startup.
"""

import os
import re
import sys
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from PIL import Image
import numpy as np
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Set offline environment variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

app = FastAPI(title="SatQueryAI SAR VQA Server (Qwen2-VL-2B)")
processor = None
model = None


def load_sar_image(image_path):
    """Load SAR image, converting GeoTIFF to 8-bit preview if needed."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                arr = src.read(1).astype(np.float32)
                dn_safe = np.where(arr > 0, arr, 1e-6)
                db = 20.0 * np.log10(dn_safe) - 83.0
                db_clipped = np.clip(db, -25.0, 0.0)
                norm = ((db_clipped - (-25.0)) / 25.0 * 255.0).astype(np.uint8)
                return Image.fromarray(norm).convert("RGB")
        except Exception:
            pass
    return Image.open(image_path).convert("RGB")


class VQARequest(BaseModel):
    image_path: str
    question: Optional[str] = None
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = 256


@app.on_event("startup")
def startup_event():
    global processor, model
    print("[SERVER 8002] Loading Qwen2-VL-2B model once at startup...")
    t0 = time.time()
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
    dur = time.time() - t0
    print(f"[SERVER 8002] Qwen2-VL-2B model loaded successfully in {dur:.2f}s. Server ready.")


@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "loading",
        "model_loaded": model is not None,
        "model": "Qwen2-VL-2B",
        "port": 8002,
    }


@app.post("/vqa")
def vqa(req: VQARequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="SAR model is not loaded yet")
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"Image file not found: {req.image_path}")

    prompt_text = req.question or req.prompt or "Describe this SAR image in detail, noting backscatter patterns, surface roughness, structural outlines, and potential water or urban features."

    t0 = time.time()
    pil_img = load_sar_image(req.image_path)
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

    max_tok = req.max_new_tokens or 256
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tok)
    torch.cuda.synchronize()
    infer_time = time.time() - t0

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

    return {
        "success": True,
        "image_path": req.image_path,
        "prompt": prompt_text,
        "answer": cleaned_output,
        "load_time_s": 0.0,
        "inference_time_s": round(infer_time, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
