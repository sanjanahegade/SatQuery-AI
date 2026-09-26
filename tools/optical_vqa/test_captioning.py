"""Test script for caption_image() function using GeoChat-7B."""

import os
import sys
import time

TOOLS_OPTICAL = r"D:\satQai\tools\optical_vqa"
if TOOLS_OPTICAL not in sys.path:
    sys.path.insert(0, TOOLS_OPTICAL)

from optical_vqa import get_engine, caption_image

print("=" * 80)
print("GEOCHAT-7B OPTICAL SCENE CAPTIONING TEST")
print("=" * 80)

# Step 1: Model Loading (once)
t0 = time.time()
engine = get_engine()
load_time = time.time() - t0
print(f"GeoChat-7B loaded in {load_time:.2f}s.\n")

# Step 2: Test 1 - Real Sentinel-2 GeoTIFF
s2_path = r"D:\satQai\data\real_test\sentinel2_mysuru.tif"
print("-" * 80)
print(f"TEST 1: Real Sentinel-2 Image ({s2_path})")
print("Prompt: 'Describe the land-cover and major objects visible in this image.'")
t1 = time.time()
caption_s2 = caption_image(s2_path, engine=engine)
time_s2 = time.time() - t1
print(f"Caption generated in {time_s2:.2f}s:")
print(f"OUTPUT:\n{caption_s2}\n")

# Step 3: Test 2 - Existing GeoChat optical test image
test_img_path = r"D:\satQai\GeoChat\images\scene.jpg"
print("-" * 80)
print(f"TEST 2: GeoChat Optical Test Image ({test_img_path})")
print("Prompt: 'Describe the land-cover and major objects visible in this image.'")
t2 = time.time()
caption_test = caption_image(test_img_path, engine=engine)
time_test = time.time() - t2
print(f"Caption generated in {time_test:.2f}s:")
print(f"OUTPUT:\n{caption_test}\n")

print("=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 80)
