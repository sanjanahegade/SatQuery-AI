"""Run Bi-temporal Change-VQA across all three Sentinel-2 pairs."""

import os
import sys
import json
import time

TOOLS_CHANGE = r"D:\satQai\tools\change_vqa"
if TOOLS_CHANGE not in sys.path:
    sys.path.insert(0, TOOLS_CHANGE)

from change_vqa import analyze_change_pair
from optical_vqa import get_engine

DATA_DIR = r"D:\satQai\data\real_test"
REPORT_PATH = os.path.join(DATA_DIR, "change_vqa_report.json")

print("=" * 80)
print("BI-TEMPORAL CHANGE-VQA EXECUTION (3 SATELLITE PAIRS)")
print("=" * 80)

# 1. Load GeoChat-7B once
print("Initializing GeoChat-7B engine in 4-bit mode on GPU...")
t_start = time.time()
engine = get_engine()
print(f"Engine initialized in {time.time() - t_start:.2f}s.\n")

pairs_config = [
    {
        "name": "Pair A - Mysuru Seasonal (May 2026 vs Aug 2026)",
        "pre": os.path.join(DATA_DIR, "sentinel2_mysuru_earlier.tif"),
        "post": os.path.join(DATA_DIR, "sentinel2_mysuru.tif"),
    },
    {
        "name": "Pair B - Nepal Flash Flood (Aug 12 Pre vs Aug 27 Post)",
        "pre": os.path.join(DATA_DIR, "sentinel2_nepal_pre_flood.tif"),
        "post": os.path.join(DATA_DIR, "sentinel2_nepal_post_flood.tif"),
    },
    {
        "name": "Pair C - Jaipur Urban Seasonal (May 2026 vs Sep 2026)",
        "pre": os.path.join(DATA_DIR, "sentinel2_jaipur_earlier.tif"),
        "post": os.path.join(DATA_DIR, "sentinel2_jaipur_later.tif"),
    },
]

results = []

for cfg in pairs_config:
    res = analyze_change_pair(
        pre_path=cfg["pre"],
        post_path=cfg["post"],
        pair_name=cfg["name"],
        out_dir=DATA_DIR,
        engine=engine,
    )
    results.append(res)

# Save JSON report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved comprehensive Change-VQA report to {REPORT_PATH}")

# Print summary deliverables
print("\n" + "=" * 80)
print("FINAL CHANGE-VQA DELIVERABLES FOR ALL 3 PAIRS")
print("=" * 80)

for r in results:
    print(f"\n>>> {r['pair_name']} <<<")
    print(f"[REGISTRATION]: {r['registration_info']['resampling_method']} on {r['registration_info']['target_dimensions']} grid")
    print(f"[NORMALIZATION]: {r['diff_stats']['normalization_formula']}")
    print(f"\n[1. BEFORE CAPTION]:\n{r['before_caption']}")
    print(f"\n[2. AFTER CAPTION]:\n{r['after_caption']}")
    print(f"\n[3. PIXEL DIFFERENCE RESULT]:\n{r['pixel_difference_result']}")
    print(f"\n[4. COMBINED CHANGE INTERPRETATION]:\n{r['combined_change_interpretation']}")
    print(f"\n[SAVED ARTIFACTS]:")
    print(f"  Difference Map PNG: {r['change_map_png']}")
    print(f"  Raw Difference NPY: {r['raw_diff_npy']}")
    print("-" * 80)
