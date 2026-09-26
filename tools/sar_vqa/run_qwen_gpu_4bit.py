"""GPU-based SAR VQA test using Qwen2-VL-2B-Instruct in 4-bit quantization.

STEP 2: Load Qwen2-VL-2B-Instruct in 4-bit on GPU using BitsAndBytesConfig
STEP 3: Run SAR VQA test against Sentinel-1 Mysuru image
STEP 4: Measure timing (load time, inference time), peak VRAM, and report raw output
"""

import os
import sys
import time
import traceback
import psutil
import torch
from PIL import Image

def get_vram_info():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        return {
            "allocated_mb": round(alloc, 2),
            "reserved_mb": round(reserved, 2),
            "peak_allocated_mb": round(max_alloc, 2),
            "peak_reserved_mb": round(max_reserved, 2),
        }
    return {}

print("=" * 80)
print("GPU Qwen2-VL-2B-Instruct 4-BIT (NF4) SAR VQA TEST")
print("=" * 80)

# 1. Environment & Hardware Audit
print("\n[STEP 1: ENVIRONMENT & HARDWARE AUDIT]")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("FATAL: CUDA is not available in PyTorch. Cannot proceed with GPU test.")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
print(f"GPU Device: {device_name}")
print(f"Total VRAM: {total_vram_gb} GB")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes Version: {bnb.__version__}")
except ImportError as e:
    print(f"FATAL: bitsandbytes failed to import: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
except ImportError as e:
    print(f"FATAL: transformers failed to import: {e}")
    sys.exit(1)

# 2. SAR Image Preprocessing / Verification
print("\n[STEP 2: SAR IMAGE VERIFICATION & PREPARATION]")
preview_path = r"D:\satQai\data\real_test\sentinel1_mysuru_vv_preview.png"
raw_tif_path = r"D:\satQai\data\real_test\sentinel1_mysuru_vv.tif"

if not os.path.exists(preview_path):
    print(f"Preview PNG not found at {preview_path}. Generating from raw GeoTIFF...")
    import rasterio
    import numpy as np
    
    if not os.path.exists(raw_tif_path):
        print(f"FATAL: Raw SAR GeoTIFF not found at {raw_tif_path}")
        sys.exit(1)
        
    with rasterio.open(raw_tif_path) as src:
        arr = src.read(1).astype(np.float32)
        
    db = 20.0 * np.log10(np.maximum(arr, 1.0)) - 83.0
    p1 = float(np.percentile(db, 1.0))
    p99 = float(np.percentile(db, 99.0))
    db_clipped = np.clip(db, p1, p99)
    norm_uint8 = ((db_clipped - p1) / (p99 - p1 + 1e-6) * 255.0).astype(np.uint8)
    rgb_arr = np.stack([norm_uint8, norm_uint8, norm_uint8], axis=-1)
    img = Image.fromarray(rgb_arr)
    img.save(preview_path)
    print(f"Saved generated preview PNG to {preview_path}")

pil_image = Image.open(preview_path)
print(f"Loaded preview image: {preview_path}")
print(f"Dimensions: {pil_image.size}, Mode: {pil_image.mode}")

# 3. Model Loading in 4-bit on GPU
print("\n[STEP 3: LOADING QWEN2-VL-2B-INSTRUCT IN 4-BIT ON GPU]")
model_id = "Qwen/Qwen2-VL-2B-Instruct"

from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
load_start = time.time()

try:
    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f"Loading model with 4-bit BitsAndBytesConfig (device_map='cuda')...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    load_time = time.time() - load_start
    print(f"SUCCESS: Model loaded in {load_time:.2f} seconds.")
except Exception as e:
    load_time = time.time() - load_start
    print(f"\nFAILED: Model loading encountered an exception after {load_time:.2f}s:")
    traceback.print_exc()
    print("\n" + "=" * 80)
    print("FINAL TEST REPORT")
    print("=" * 80)
    print(f"Status: FAILED_DURING_LOAD")
    print(f"Load Time: {load_time:.2f}s")
    print(f"Peak VRAM: {get_vram_info()}")
    print(f"Error: {e}")
    print("=" * 80)
    sys.exit(1)

vram_after_load = get_vram_info()
print(f"VRAM Allocated after load: {vram_after_load['allocated_mb']} MB")
print(f"VRAM Reserved after load: {vram_after_load['reserved_mb']} MB")
print(f"Model primary device: {next(model.parameters()).device}")

# 4. Inference Execution
print("\n[STEP 4: INFERENCE EXECUTION ON GPU]")
prompt_text = "Describe this SAR image."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt_text},
        ],
    }
]

prep_start = time.time()
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
inputs = inputs.to("cuda")
prep_time = time.time() - prep_start
print(f"Input processing & tokenization took {prep_time:.2f}s. Input sequence length: {inputs.input_ids.shape[1]}")

print("Generating response on GPU (max_new_tokens=256)...")
infer_start = time.time()

try:
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256
        )
    torch.cuda.synchronize()
    infer_time = time.time() - infer_start
    print(f"Inference completed in {infer_time:.2f} seconds.")
except Exception as e:
    infer_time = time.time() - infer_start
    print(f"\nFAILED: Inference encountered an exception after {infer_time:.2f}s:")
    traceback.print_exc()
    print("\n" + "=" * 80)
    print("FINAL TEST REPORT")
    print("=" * 80)
    print(f"Status: FAILED_DURING_INFERENCE")
    print(f"Load Time: {load_time:.2f}s")
    print(f"Inference Time: {infer_time:.2f}s")
    print(f"Peak VRAM: {get_vram_info()}")
    print(f"Error: {e}")
    print("=" * 80)
    sys.exit(1)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

vram_final = get_vram_info()

print("\n" + "=" * 80)
print("FINAL TEST REPORT (GPU 4-BIT SAR VQA)")
print("=" * 80)
print(f"Model ID: {model_id}")
print(f"Quantization: 4-bit NF4 (BitsAndBytes)")
print(f"Device: {device_name} (device_map='cuda')")
print(f"Load Time: {load_time:.2f} s")
print(f"Inference Time: {infer_time:.2f} s")
print(f"Peak VRAM Allocated: {vram_final['peak_allocated_mb']} MB ({round(vram_final['peak_allocated_mb']/1024, 2)} GB)")
print(f"Peak VRAM Reserved: {vram_final['peak_reserved_mb']} MB ({round(vram_final['peak_reserved_mb']/1024, 2)} GB)")
print(f"Total Available VRAM: {total_vram_gb} GB")
print(f"VRAM Headroom: {round(total_vram_gb - vram_final['peak_reserved_mb']/1024, 2)} GB")
print("-" * 80)
print("RAW MODEL OUTPUT:")
print(output_text)
print("-" * 80)

if infer_time > 120.0:
    print("WARNING: Inference took over 2 minutes - potential silent CPU fallback or extreme tokenization!")
    print("Status: SLOW_INFERENCE")
else:
    print("Status: SUCCESS (Fast GPU inference confirmed)")
print("=" * 80)
