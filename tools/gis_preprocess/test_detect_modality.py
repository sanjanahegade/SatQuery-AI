"""Generate synthetic GeoTIFF images and test detect_modality().

Tests 5 cases:
1. Synthetic Optical (3-band uint8, distinct channel means, Sentinel-2 tag)
2. Synthetic SAR (1-band float32, speckle noise, Sentinel-1 tag, VV description)
3. Synthetic Ambiguous/Unknown (3-band uint8 grayscale, no tags)
4. Synthetic Optical without metadata (3-band uint8, distinct channel variance, no tags)
5. Synthetic Optical with Heavy Cloud Skew (3-band uint16, engineered skewness 4.0-5.0, no tags)
"""

import os
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.stats import skew

# Ensure import of detect_modality
from detect_modality import detect_modality


def create_synthetic_images(output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    width, height = 256, 256
    transform = from_origin(500000, 3000000, 10, 10)
    crs = "EPSG:32643"

    np.random.seed(42)

    # -------------------------------------------------------------------------
    # 1. Synthetic Optical (3-band uint8, distinct R/G/B, Sentinel-2 tag)
    # -------------------------------------------------------------------------
    path_optical = os.path.join(output_dir, "synthetic_optical.tif")
    r = np.clip(np.random.normal(150, 15, (height, width)), 0, 255).astype(np.uint8)
    g = np.clip(np.random.normal(95, 12, (height, width)), 0, 255).astype(np.uint8)
    b = np.clip(np.random.normal(50, 10, (height, width)), 0, 255).astype(np.uint8)
    optical_data = np.stack([r, g, b])

    with rasterio.open(
        path_optical,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=optical_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(optical_data)
        dst.descriptions = ("Red (B4)", "Green (B3)", "Blue (B2)")
        dst.update_tags(sensor="Sentinel-2", mission="MSI")

    paths["optical"] = path_optical

    # -------------------------------------------------------------------------
    # 2. Synthetic SAR (1-band float32, speckle noise, Sentinel-1, VV)
    # -------------------------------------------------------------------------
    path_sar = os.path.join(output_dir, "synthetic_sar.tif")
    # Multiplicative speckle noise (Gamma/exponential distributed intensity)
    speckle_noise = np.random.exponential(scale=1.0, size=(height, width))
    base_intensity = 40.0 + 20.0 * np.sin(np.linspace(0, 3.14, height))[:, None]
    sar_data = (base_intensity * speckle_noise).astype(np.float32)[np.newaxis, ...]

    with rasterio.open(
        path_sar,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=sar_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(sar_data)
        dst.descriptions = ("VV",)
        dst.update_tags(sensor="Sentinel-1", polarization="VV", acquisition_mode="IW")

    paths["sar"] = path_sar

    # -------------------------------------------------------------------------
    # 3. Synthetic Ambiguous/Unknown (3-band uint8 grayscale, no tags)
    # -------------------------------------------------------------------------
    path_ambiguous = os.path.join(output_dir, "synthetic_ambiguous.tif")
    # Exactly identical channels (grayscale), smooth variation, no metadata
    gray = np.clip(np.random.normal(120, 15, (height, width)), 0, 255).astype(np.uint8)
    ambiguous_data = np.stack([gray, gray, gray])

    with rasterio.open(
        path_ambiguous,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=ambiguous_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(ambiguous_data)
        # No descriptions, no tags

    paths["ambiguous"] = path_ambiguous

    # -------------------------------------------------------------------------
    # 4. Synthetic Optical without metadata (3-band uint8, distinct variance)
    # -------------------------------------------------------------------------
    path_optical_no_meta = os.path.join(output_dir, "synthetic_optical_no_meta.tif")
    r_nometa = np.clip(np.random.normal(165, 18, (height, width)), 0, 255).astype(np.uint8)
    g_nometa = np.clip(np.random.normal(110, 14, (height, width)), 0, 255).astype(np.uint8)
    b_nometa = np.clip(np.random.normal(60, 12, (height, width)), 0, 255).astype(np.uint8)
    optical_nometa_data = np.stack([r_nometa, g_nometa, b_nometa])

    with rasterio.open(
        path_optical_no_meta,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=optical_nometa_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(optical_nometa_data)
        # No sensor tags, no band descriptions

    paths["optical_no_meta"] = path_optical_no_meta

    # -------------------------------------------------------------------------
    # 5. Synthetic Optical with Heavy Cloud Skew (3-band uint16, skew ~4.5, no tags)
    # -------------------------------------------------------------------------
    path_heavy_cloud = os.path.join(output_dir, "synthetic_optical_heavy_cloud_skew.tif")
    # Background: surface reflectance uint16 (distinct channel means/variance)
    r_cloud = np.random.normal(1200, 150, (height, width))
    g_cloud = np.random.normal(1600, 180, (height, width))
    b_cloud = np.random.normal(800, 100, (height, width))

    # Add compact cloud puffs covering ~4.2% of pixels with high brightness (~14500 DN)
    # yielding an engineered sample skewness in the 4.0 - 5.0 range
    cloud_mask = np.zeros((height, width), dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    for cy, cx in [(80, 80), (170, 160)]:
        cloud_mask |= ((yy - cy)**2 + (xx - cx)**2 < 21**2)

    cloud_count = int(np.sum(cloud_mask))
    r_cloud[cloud_mask] = 14500 + np.random.normal(0, 300, cloud_count)
    g_cloud[cloud_mask] = 14500 + np.random.normal(0, 300, cloud_count)
    b_cloud[cloud_mask] = 14500 + np.random.normal(0, 300, cloud_count)

    cloud_data = np.clip(np.stack([r_cloud, g_cloud, b_cloud]), 0, 65535).astype(np.uint16)

    with rasterio.open(
        path_heavy_cloud,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=cloud_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(cloud_data)
        # No sensor tags, no band descriptions (tests pixel statistics alone)

    paths["optical_heavy_cloud_skew"] = path_heavy_cloud

    return paths


def test_synthetic_optical_heavy_cloud_skew(image_path: str = None) -> dict:
    """Validate that a high-skewness optical image safely degrades to 'unknown'
    or classifies as 'optical', and NEVER falsely classifies as 'sar' or receives SAR credit.
    """
    if image_path is None:
        output_dir = r"D:\satQai\data\synthetic_test"
        paths = create_synthetic_images(output_dir)
        image_path = paths["optical_heavy_cloud_skew"]

    result = detect_modality(image_path)
    # Evaluation assertions:
    assert result["modality"] != "sar", f"Failure: high-skew optical misclassified as SAR! {result}"
    assert result["sar_score"] == 0.0, f"Failure: false SAR credit awarded! {result}"
    assert result["modality"] in ("optical", "unknown"), f"Unexpected modality: {result['modality']}"
    return result


def main():
    output_dir = r"D:\satQai\data\synthetic_test"
    print(f"Creating synthetic test GeoTIFFs in {output_dir}...")
    paths = create_synthetic_images(output_dir)

    print("\n" + "=" * 70)
    print("RUNNING MODALITY DETECTION ON SYNTHETIC TEST CASES")
    print("=" * 70)

    test_labels = [
        ("1. Synthetic 'optical' (Sentinel-2, 3-band RGB uint8)", paths["optical"]),
        ("2. Synthetic 'SAR' (Sentinel-1 VV, 1-band float32 speckle)", paths["sar"]),
        ("3. Synthetic 'ambiguous/unknown' (3-band grayscale, no tags)", paths["ambiguous"]),
        ("4. Synthetic 'optical without metadata' (3-band uint8, clear variance, no tags)", paths["optical_no_meta"]),
        ("5. Synthetic 'optical heavy cloud skew' (3-band uint16, skew ~4.5, no tags)", paths["optical_heavy_cloud_skew"]),
    ]

    all_results = {}
    for label, path in test_labels:
        print(f"\n>>> TEST CASE: {label}")
        print(f"    File: {path}")
        result = detect_modality(path)
        all_results[label] = result
        print(json.dumps(result, indent=2))

    print("\nRunning assertion check on boundary test case 5...")
    test_synthetic_optical_heavy_cloud_skew(paths["optical_heavy_cloud_skew"])
    print("Assertion passed: boundary test case 5 safely evaluated without false SAR classification!")

    return all_results


if __name__ == "__main__":
    main()
