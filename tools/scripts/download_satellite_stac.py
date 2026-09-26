"""Download real Sentinel-2 and Sentinel-1 scenes over Mysuru from Planetary Computer STAC
and validate detect_modality() against them.
"""

import os
import json
import time
from typing import Any, Dict

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy.interpolate import LinearNDInterpolator
import pystac_client
import planetary_computer as pc

# Import the modality detector from gis_preprocess
import sys
sys.path.insert(0, r"D:\satQai\tools\gis_preprocess")
from detect_modality import detect_modality

STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [76.55, 12.25, 76.70, 12.35]
DATETIME_RANGE = "2026-08-18/2026-09-17"
OUT_DIR = r"D:\satQai\data\real_test"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report: Dict[str, Any] = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": {
            "stac_endpoint": STAC_ENDPOINT,
            "bbox": BBOX,
            "datetime_range": DATETIME_RANGE,
            "target_directory": OUT_DIR
        },
        "stop_conditions_hit": [],
        "sentinel2": {},
        "sentinel1": {}
    }

    print("=" * 75)
    print("STEP 1: QUERYING PLANETARY COMPUTER STAC API")
    print("=" * 75)

    client = pystac_client.Client.open(STAC_ENDPOINT, modifier=pc.sign_inplace)

    # -------------------------------------------------------------------------
    # Sentinel-2 Search
    # -------------------------------------------------------------------------
    print(f"\nSearching Sentinel-2 (sentinel-2-l2a) in bbox {BBOX} for {DATETIME_RANGE}...")
    s2_search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATETIME_RANGE,
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        limit=1
    )
    s2_items = list(s2_search.items())
    if not s2_items:
        stop_msg = f"STOP: No Sentinel-2 scene found in exact bbox {BBOX} and date range {DATETIME_RANGE}."
        print(f"\n[!] {stop_msg}")
        report["stop_conditions_hit"].append(stop_msg)
        with open(os.path.join(OUT_DIR, "validation_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return

    s2_item = s2_items[0]
    print(f"-> Selected Sentinel-2 item: {s2_item.id}")
    print(f"   Acquisition Date: {s2_item.datetime}")
    print(f"   Cloud Cover: {s2_item.properties.get('eo:cloud_cover'):.2f}%")

    # -------------------------------------------------------------------------
    # Sentinel-1 Search
    # -------------------------------------------------------------------------
    print(f"\nSearching Sentinel-1 (sentinel-1-grd) in bbox {BBOX} for {DATETIME_RANGE}...")
    s1_search = client.search(
        collections=["sentinel-1-grd"],
        bbox=BBOX,
        datetime=DATETIME_RANGE,
        limit=10
    )
    s1_items = list(s1_search.items())
    if not s1_items:
        stop_msg = f"STOP: No Sentinel-1 scene found in exact bbox {BBOX} and date range {DATETIME_RANGE}."
        print(f"\n[!] {stop_msg}")
        report["stop_conditions_hit"].append(stop_msg)
        with open(os.path.join(OUT_DIR, "validation_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return

    # Check for IW mode and VV polarization
    selected_s1_item = None
    vv_asset_key = None
    for item in s1_items:
        props = item.properties
        mode = (props.get("sar:instrument_mode") or props.get("sar:mode") or props.get("instrument_mode"))
        pols = props.get("sar:polarizations") or []
        # Find asset key matching VV
        vv_key = None
        for k in item.assets.keys():
            if k.lower() == "vv":
                vv_key = k
                break
        
        print(f"   Inspecting S1 item {item.id}: mode={mode}, polarizations={pols}, vv_asset={vv_key}")
        if mode == "IW" and ("VV" in [p.upper() for p in pols]) and vv_key:
            selected_s1_item = item
            vv_asset_key = vv_key
            break

    if not selected_s1_item:
        stop_msg = f"STOP: No Sentinel-1 scene found with BOTH IW mode and VV asset in exact search window."
        print(f"\n[!] {stop_msg}")
        report["stop_conditions_hit"].append(stop_msg)
        with open(os.path.join(OUT_DIR, "validation_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return

    print(f"-> Selected Sentinel-1 item: {selected_s1_item.id}")
    print(f"   Acquisition Mode: {selected_s1_item.properties.get('sar:instrument_mode')}")
    print(f"   VV Asset Key: {vv_asset_key}")

    # -------------------------------------------------------------------------
    # STEP 2: DOWNLOAD & STACK SENTINEL-2 (B04, B03, B02)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 2: DOWNLOADING & STACKING SENTINEL-2 BANDS (B04, B03, B02)")
    print("=" * 75)

    s2_out_path = os.path.join(OUT_DIR, "sentinel2_mysuru.tif")
    b04_url = pc.sign(s2_item.assets["B04"].href)
    b03_url = pc.sign(s2_item.assets["B03"].href)
    b02_url = pc.sign(s2_item.assets["B02"].href)

    print("Opening remote Sentinel-2 B04 to compute Mysuru spatial window...")
    with rasterio.open(b04_url) as src_b4:
        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src_b4.crs, *BBOX)
        window = from_bounds(minx, miny, maxx, maxy, transform=src_b4.transform).round_offsets().round_lengths()
        out_transform = src_b4.window_transform(window)
        out_crs = src_b4.crs
        out_meta = src_b4.meta.copy()
        orig_s2_tags = src_b4.tags()
        
        print(f"   Window computed: {window}")
        print("   Reading Band 4 (Red)...")
        b4_data = src_b4.read(1, window=window)

    print("   Reading Band 3 (Green)...")
    with rasterio.open(b03_url) as src_b3:
        b3_data = src_b3.read(1, window=window)

    print("   Reading Band 2 (Blue)...")
    with rasterio.open(b02_url) as src_b2:
        b2_data = src_b2.read(1, window=window)

    s2_stack = np.stack([b4_data, b3_data, b2_data])  # Shape: (3, H, W)
    print(f"   Stacked array shape: {s2_stack.shape}, dtype: {s2_stack.dtype}")

    # Write stacked GeoTIFF preserving original metadata
    out_meta.update({
        "count": 3,
        "height": window.height,
        "width": window.width,
        "transform": out_transform,
        "crs": out_crs,
        "dtype": s2_stack.dtype,
        "driver": "GTiff"
    })

    print(f"Writing stacked image to: {s2_out_path}")
    with rasterio.open(s2_out_path, "w", **out_meta) as dst:
        dst.write(s2_stack)
        # Preserve original source tags (e.g. AREA_OR_POINT)
        if orig_s2_tags:
            dst.update_tags(**orig_s2_tags)
        # CRITICAL: We do NOT add synthetic sensor="Sentinel-2" or band descriptions

    # -------------------------------------------------------------------------
    # STEP 3: DOWNLOAD SENTINEL-1 VV
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3: DOWNLOADING SENTINEL-1 VV (MYSURU REGION)")
    print("=" * 75)

    s1_out_path = os.path.join(OUT_DIR, "sentinel1_mysuru_vv.tif")
    vv_url = pc.sign(selected_s1_item.assets[vv_asset_key].href)

    print(f"Opening remote Sentinel-1 VV COG ({selected_s1_item.id})...")
    with rasterio.open(vv_url) as src_s1:
        gcps, gcp_crs = src_s1.gcps
        orig_s1_tags = src_s1.tags()
        orig_s1_dtype = src_s1.dtypes[0]
        
        # Calculate pixel window corresponding to Mysuru bbox using GCP interpolation
        lons = [g.x for g in gcps]
        lats = [g.y for g in gcps]
        rows = [g.row for g in gcps]
        cols = [g.col for g in gcps]
        interp_r = LinearNDInterpolator(list(zip(lons, lats)), rows)
        interp_c = LinearNDInterpolator(list(zip(lons, lats)), cols)
        
        corners_lon = [BBOX[0], BBOX[2], BBOX[2], BBOX[0]]
        corners_lat = [BBOX[1], BBOX[1], BBOX[3], BBOX[3]]
        r_pts = interp_r(corners_lon, corners_lat)
        c_pts = interp_c(corners_lon, corners_lat)
        
        row_min = int(max(0, np.floor(np.min(r_pts))))
        row_max = int(min(src_s1.height, np.ceil(np.max(r_pts))))
        col_min = int(max(0, np.floor(np.min(c_pts))))
        col_max = int(min(src_s1.width, np.ceil(np.max(c_pts))))
        
        s1_window = rasterio.windows.Window(col_min, row_min, col_max - col_min, row_max - row_min)
        print(f"   S1 Mysuru pixel window: {s1_window}")
        print("   Reading Sentinel-1 VV window...")
        s1_data = src_s1.read(1, window=s1_window)
        print(f"   Read S1 shape: {s1_data.shape}, dtype: {s1_data.dtype}")

        # Filter GCPs that fall within this spatial window and adjust coordinates
        subset_gcps = []
        for g in gcps:
            if row_min <= g.row <= row_max and col_min <= g.col <= col_max:
                from rasterio.control import GroundControlPoint
                subset_gcps.append(GroundControlPoint(
                    row=g.row - row_min,
                    col=g.col - col_min,
                    x=g.x,
                    y=g.y,
                    z=g.z,
                    id=g.id
                ))

        # Write to local GeoTIFF
        s1_meta = {
            "driver": "GTiff",
            "count": 1,
            "dtype": s1_data.dtype,
            "width": s1_window.width,
            "height": s1_window.height,
        }

    print(f"Writing Sentinel-1 VV image to: {s1_out_path}")
    with rasterio.open(s1_out_path, "w", **s1_meta) as dst_s1:
        dst_s1.write(s1_data, 1)
        # Preserve original source tags naturally present in the product (e.g. TIFFTAG_IMAGEDESCRIPTION)
        if orig_s1_tags:
            dst_s1.update_tags(**orig_s1_tags)
        if subset_gcps:
            dst_s1.gcps = (subset_gcps, gcp_crs)
        # CRITICAL: We do NOT add synthetic sensor="Sentinel-1" or band description="VV"

    # -------------------------------------------------------------------------
    # STEP 4: GATHER RASTER METADATA & VERIFY FILES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 4: VERIFYING DOWNLOADED FILES & GATHERING METADATA")
    print("=" * 75)

    def inspect_file(filepath: str) -> Dict[str, Any]:
        size_bytes = os.path.getsize(filepath)
        with rasterio.open(filepath) as ds:
            arr = ds.read()
            min_vals = [float(np.min(arr[b])) for b in range(ds.count)]
            max_vals = [float(np.max(arr[b])) for b in range(ds.count)]
            return {
                "file_path": filepath,
                "file_size_bytes": size_bytes,
                "file_size_mb": round(size_bytes / (1024 * 1024), 2),
                "is_non_empty": size_bytes > 0,
                "opens_cleanly": True,
                "driver": ds.driver,
                "count": ds.count,
                "dtype": str(ds.dtypes[0]),
                "width": ds.width,
                "height": ds.height,
                "crs": str(ds.crs) if ds.crs else None,
                "transform": [float(v) for v in ds.transform],
                "descriptions": list(ds.descriptions or []),
                "tags": ds.tags(),
                "min_values": min_vals,
                "max_values": max_vals
            }

    s2_info = inspect_file(s2_out_path)
    s1_info = inspect_file(s1_out_path)

    report["sentinel2"]["stac_metadata"] = {
        "item_id": s2_item.id,
        "collection": "sentinel-2-l2a",
        "datetime": str(s2_item.datetime),
        "cloud_cover": s2_item.properties.get("eo:cloud_cover"),
        "platform": s2_item.properties.get("platform"),
        "assets_used": ["B04", "B03", "B02"]
    }
    report["sentinel2"]["raster_metadata"] = s2_info

    report["sentinel1"]["stac_metadata"] = {
        "item_id": selected_s1_item.id,
        "collection": "sentinel-1-grd",
        "datetime": str(selected_s1_item.datetime),
        "instrument_mode": selected_s1_item.properties.get("sar:instrument_mode"),
        "polarizations": selected_s1_item.properties.get("sar:polarizations"),
        "platform": selected_s1_item.properties.get("platform"),
        "asset_used": vv_asset_key
    }
    report["sentinel1"]["raster_metadata"] = s1_info

    # -------------------------------------------------------------------------
    # STEP 5: RUN MODALITY DETECTION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 5: RUNNING DETECT_MODALITY() ON REAL SATELLITE PRODUCTS")
    print("=" * 75)

    print(f"\nEvaluating Sentinel-2 product: {s2_out_path}")
    s2_detection = detect_modality(s2_out_path)
    report["sentinel2"]["detect_modality_output"] = s2_detection
    print(json.dumps(s2_detection, indent=2))

    print(f"\nEvaluating Sentinel-1 product: {s1_out_path}")
    s1_detection = detect_modality(s1_out_path)
    report["sentinel1"]["detect_modality_output"] = s1_detection
    print(json.dumps(s1_detection, indent=2))

    # -------------------------------------------------------------------------
    # STEP 6: SAVE VALIDATION REPORT
    # -------------------------------------------------------------------------
    report_path = os.path.join(OUT_DIR, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved to: {report_path}")

    # -------------------------------------------------------------------------
    # CONSOLE SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("HUMAN-READABLE VALIDATION SUMMARY")
    print("=" * 75)
    print(f"Sentinel-2 Item: {s2_item.id}")
    print(f"  File: {s2_out_path} ({s2_info['file_size_mb']} MB, {s2_info['width']}x{s2_info['height']}, {s2_info['count']} bands, {s2_info['dtype']})")
    print(f"  Detected Modality: {s2_detection['modality'].upper()} (confidence: {s2_detection['confidence']})")
    print(f"  Key Evidence: {s2_detection['evidence'][-1]}")
    
    print(f"\nSentinel-1 Item: {selected_s1_item.id}")
    print(f"  File: {s1_out_path} ({s1_info['file_size_mb']} MB, {s1_info['width']}x{s1_info['height']}, {s1_info['count']} band, {s1_info['dtype']})")
    print(f"  Detected Modality: {s1_detection['modality'].upper()} (confidence: {s1_detection['confidence']})")
    print(f"  Key Evidence: {s1_detection['evidence'][-1]}")
    print("=" * 75)


if __name__ == "__main__":
    main()
