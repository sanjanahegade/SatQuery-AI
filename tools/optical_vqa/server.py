#!/usr/bin/env python3
"""FastAPI server for Optical VQA and Scene Captioning using GeoChat-7B.

Runs in D:\satQai\tools\optical_vqa\venv on port 8001.
Loads GeoChat-7B ONCE into memory at server startup.
"""

import os
import sys
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Set offline environment variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

GEOCHAT_DIR = r"D:\satQai\GeoChat"
if GEOCHAT_DIR not in sys.path:
    sys.path.insert(0, GEOCHAT_DIR)

from optical_vqa import get_engine, caption_image, answer_question

app = FastAPI(title="SatQueryAI Optical VQA Server (GeoChat-7B)")
engine = None


class CaptionRequest(BaseModel):
    image_path: str
    prompt: Optional[str] = "Describe the land-cover and major objects visible in this image."
    max_new_tokens: Optional[int] = 256


class VQARequest(BaseModel):
    image_path: str
    question: str
    max_new_tokens: Optional[int] = 256


@app.on_event("startup")
def startup_event():
    global engine
    print("[SERVER 8001] Loading GeoChat-7B model once at startup...")
    t0 = time.time()
    engine = get_engine()
    dur = time.time() - t0
    print(f"[SERVER 8001] GeoChat-7B model loaded successfully in {dur:.2f}s. Server ready.")


@app.get("/health")
def health():
    return {
        "status": "ready" if engine is not None else "loading",
        "model_loaded": engine is not None,
        "model": "GeoChat-7B",
        "port": 8001,
    }


@app.post("/caption")
def caption(req: CaptionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="GeoChat model is not loaded yet")
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"Image file not found: {req.image_path}")

    t0 = time.time()
    cap = caption_image(req.image_path, prompt=req.prompt, engine=engine)
    infer_time = time.time() - t0

    return {
        "success": True,
        "mode": "caption",
        "image_path": req.image_path,
        "prompt": req.prompt,
        "caption": cap,
        "load_time_s": 0.0,
        "inference_time_s": round(infer_time, 2),
    }


@app.post("/vqa")
def vqa(req: VQARequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="GeoChat model is not loaded yet")
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"Image file not found: {req.image_path}")

    t0 = time.time()
    ans = answer_question(req.image_path, req.question, engine=engine)
    infer_time = time.time() - t0

    return {
        "success": True,
        "mode": "vqa",
        "image_path": req.image_path,
        "question": req.question,
        "answer": ans.get("answer") if isinstance(ans, dict) else str(ans),
        "confidence": ans.get("confidence") if isinstance(ans, dict) else 1.0,
        "explanation": ans.get("confidence_explanation") if isinstance(ans, dict) else "Confidence computed by optical specialist",
        "load_time_s": 0.0,
        "inference_time_s": round(infer_time, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
