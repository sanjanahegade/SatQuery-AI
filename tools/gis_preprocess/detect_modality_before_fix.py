"""Modality detection for satellite imagery (Optical vs. SAR vs. Unknown).

Implements multi-check weighted evidence accumulation, an absolute evidence floor,
and calibrated normalized confidence with evidence-completeness scaling.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Theoretical maximum evidence achievable for each modality class
OPTICAL_MAX_EVIDENCE = 0.95  # metadata(0.50) + band/dtype(0.20) + channel(0.15) + value(0.10)
SAR_MAX_EVIDENCE = 0.85      # metadata(0.50) + band/dtype(0.20) + value(0.10) + texture(0.05)


def _compute_skewness(arr: np.ndarray) -> float:
    """Compute sample skewness with fallback if scipy is not available."""
    try:
        from scipy.stats import skew
        return float(skew(arr, axis=None, nan_policy='omit'))
    except Exception:
        flat = arr[np.isfinite(arr)].astype(np.float64)
        if flat.size < 3:
            return 0.0
        mean = np.mean(flat)
        std = np.std(flat)
        if std < 1e-7:
            return 0.0
        return float(np.mean(((flat - mean) / std) ** 3))


def detect_modality(image_path: str) -> Dict[str, Any]:
    """Detect whether a satellite image is optical, SAR, or unknown.

    Parameters
    ----------
    image_path : str
        Path to the satellite raster/image file.

    Returns
    -------
    dict
        {
            "modality": "optical" | "sar" | "unknown",
            "confidence": float (0.0 to 1.0),
            "evidence": list of str explaining which checks fired and findings
        }
    """
    if not os.path.exists(image_path):
        return {
            "modality": "unknown",
            "confidence": 0.0,
            "evidence": [f"File not found: {image_path}"]
        }

    evidence: List[str] = []
    optical_score: float = 0.0
    sar_score: float = 0.0

    # Try reading with rasterio first
    rasterio_obj = None
    try:
        import rasterio
        rasterio_obj = rasterio.open(image_path)
    except Exception as e:
        evidence.append(f"rasterio open warning: {e}")

    if rasterio_obj is None:
        # Fallback to PIL
        try:
            from PIL import Image
            pil_img = Image.open(image_path)
            arr = np.array(pil_img)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                arr = np.moveaxis(arr, -1, 0)
            count = arr.shape[0]
            dtypes = [str(arr.dtype)]
            tags_text = str(getattr(pil_img, "info", {}))
            descriptions = []
        except Exception as e2:
            return {
                "modality": "unknown",
                "confidence": 0.0,
                "evidence": [f"Failed to read image with rasterio and PIL: {e2}"]
            }
    else:
        count = rasterio_obj.count
        dtypes = list(rasterio_obj.dtypes)
        descriptions = list(rasterio_obj.descriptions or [])
        
        # Gather all tags from default and band namespaces
        all_tags = {}
        try:
            all_tags.update(rasterio_obj.tags())
        except Exception:
            pass
        for b in range(1, count + 1):
            try:
                all_tags.update(rasterio_obj.tags(b))
            except Exception:
                pass
        tags_text = " ".join(f"{k}={v}" for k, v in all_tags.items())

    # =========================================================================
    # CHECK 1 - Metadata keywords (run first, highest weight ~0.50)
    # =========================================================================
    sar_keywords_pattern = re.compile(
        r"\b(VV|VH|HH|HV|Sentinel-1|RISAT|SAR|RADAR|COSMO-SkyMed|TerraSAR)\b",
        re.IGNORECASE
    )
    optical_keywords_pattern = re.compile(
        r"\b(Sentinel-2|Cartosat|RGB|Red|Green|Blue|NIR|multispectral|Landsat|PlanetScope|WorldView)\b",
        re.IGNORECASE
    )

    desc_text = " ".join(d for d in descriptions if d)
    metadata_corpus = f"{desc_text} {tags_text}".strip()

    sar_matches = set(sar_keywords_pattern.findall(metadata_corpus))
    optical_matches = set(optical_keywords_pattern.findall(metadata_corpus))

    if sar_matches and not optical_matches:
        sar_score += 0.50
        evidence.append(
            f"CHECK 1 (Metadata): Strong SAR metadata signal {sorted(sar_matches)} (+0.50 SAR)"
        )
    elif optical_matches and not sar_matches:
        optical_score += 0.50
        evidence.append(
            f"CHECK 1 (Metadata): Strong Optical metadata signal {sorted(optical_matches)} (+0.50 Optical)"
        )
    elif optical_matches and sar_matches:
        evidence.append(
            f"CHECK 1 (Metadata): Conflicting metadata tags found (Optical: {sorted(optical_matches)}, SAR: {sorted(sar_matches)}) (0.0 credit)"
        )
    else:
        evidence.append("CHECK 1 (Metadata): No strong sensor/band metadata keywords found (0.0 credit)")

    # =========================================================================
    # Read raster pixel data (subsample to max 512x512 for fast statistics)
    # =========================================================================
    if rasterio_obj is not None:
        try:
            h, w = rasterio_obj.height, rasterio_obj.width
            max_dim = 512
            if h > max_dim or w > max_dim:
                scale = min(max_dim / h, max_dim / w)
                out_h = max(1, int(h * scale))
                out_w = max(1, int(w * scale))
                arr = rasterio_obj.read(
                    out_shape=(count, out_h, out_w),
                    resampling=rasterio.enums.Resampling.bilinear
                )
            else:
                arr = rasterio_obj.read()
        finally:
            rasterio_obj.close()

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # =========================================================================
    # CHECK 2 - Band count and dtype (weight ~0.20)
    # =========================================================================
    primary_dtype = str(dtypes[0]).lower()
    val_min, val_max = float(np.min(arr)), float(np.max(arr))
    dyn_range = val_max - val_min

    if count >= 3 and ("uint8" in primary_dtype or (val_min >= 0 and val_max <= 255.0)):
        optical_score += 0.20
        evidence.append(
            f"CHECK 2 (Band count/dtype): {count} bands, dtype={primary_dtype}, range=[{val_min:.1f}, {val_max:.1f}] typical of optical reflectance (+0.20 Optical)"
        )
    elif count == 1 and ("float" in primary_dtype or val_min < 0 or dyn_range > 5000):
        sar_score += 0.20
        evidence.append(
            f"CHECK 2 (Band count/dtype): 1 band, dtype={primary_dtype}, range=[{val_min:.1f}, {val_max:.1f}] typical of SAR amplitude/dB (+0.20 SAR)"
        )
    else:
        evidence.append(
            f"CHECK 2 (Band count/dtype): {count} band(s), dtype={primary_dtype}, range=[{val_min:.1f}, {val_max:.1f}] is inconclusive (0.0 credit)"
        )

    # =========================================================================
    # CHECK 3 - Pixel statistics (fallback when metadata is missing/inconclusive)
    # =========================================================================
    # 3a) Channel variance (weight ~0.15)
    # Only "differs -> optical" counts. Near-identical channels withhold the +0.15 bonus
    # and do NOT add anything to sar_score.
    if count >= 2:
        per_pixel_band_std = np.std(arr[:3] if count >= 3 else arr, axis=0)
        mean_band_std = float(np.mean(per_pixel_band_std))
        
        channel_means = [float(np.mean(arr[b])) for b in range(min(count, 3))]
        channel_mean_diff = float(np.ptp(channel_means))

        if mean_band_std > 2.5 or channel_mean_diff > 4.0:
            optical_score += 0.15
            evidence.append(
                f"CHECK 3a (Channel variance): Clearly differing spectral channels (mean band std={mean_band_std:.2f}, mean diff={channel_mean_diff:.2f}) (+0.15 Optical)"
            )
        else:
            evidence.append(
                f"CHECK 3a (Channel variance): Near-identical channels (mean band std={mean_band_std:.2f}) withholds +0.15 Optical bonus (0.0 SAR credit)"
            )
    else:
        evidence.append("CHECK 3a (Channel variance): Single-band image -> N/A (0.0 credit)")

    # 3b) Value distribution: skewness / negative values (weight ~0.10)
    flat_pixels = arr[0].ravel().astype(np.float64) if count == 1 else np.mean(arr[:3], axis=0).ravel().astype(np.float64)
    total_pix = flat_pixels.size
    neg_frac = float(np.sum(flat_pixels < 0.0) / max(total_pix, 1))
    skew_val = _compute_skewness(flat_pixels)

    if neg_frac > 0.05 or skew_val > 1.8:
        sar_score += 0.10
        evidence.append(
            f"CHECK 3b (Value distribution): Long-tailed / negative distribution (skewness={skew_val:.2f}, neg_frac={neg_frac*100:.1f}%) typical of SAR dB/speckle (+0.10 SAR)"
        )
    elif neg_frac <= 0.01 and abs(skew_val) < 1.2:
        optical_score += 0.10
        evidence.append(
            f"CHECK 3b (Value distribution): Non-negative, roughly Gaussian distribution (skewness={skew_val:.2f}, neg_frac={neg_frac*100:.1f}%) typical of optical reflectance (+0.10 Optical)"
        )
    else:
        evidence.append(
            f"CHECK 3b (Value distribution): Distribution inconclusive (skewness={skew_val:.2f}, neg_frac={neg_frac*100:.1f}%) (0.0 credit)"
        )

    # 3c) Texture / speckle proxy: local variance vs global variance (weight ~0.05)
    # High grainy local variance relative to global variance -> WEAK supporting evidence for SAR only.
    # Capped so it can NEVER push a decision alone (requires sar_score > 0 prior).
    intensity_2d = arr[0].astype(np.float32) if count == 1 else np.mean(arr[:3], axis=0).astype(np.float32)
    h_2d, w_2d = intensity_2d.shape
    block_size = 8
    h_b, w_b = h_2d // block_size, w_2d // block_size

    has_speckle = False
    median_cv = 0.0
    if h_b > 0 and w_b > 0:
        blocks = intensity_2d[:h_b * block_size, :w_b * block_size].reshape(h_b, block_size, w_b, block_size)
        block_stds = np.std(blocks, axis=(1, 3))
        block_means = np.abs(np.mean(blocks, axis=(1, 3)))
        valid_mask = block_means > 1e-4
        if np.any(valid_mask):
            cvs = block_stds[valid_mask] / (block_means[valid_mask] + 1e-6)
            median_cv = float(np.median(cvs))
            if median_cv > 0.28:
                has_speckle = True

    if has_speckle:
        if sar_score > 0.0:
            sar_score += 0.05
            evidence.append(
                f"CHECK 3c (Texture/speckle): High local grainy speckle (median Cv={median_cv:.2f}) supported by other SAR evidence (+0.05 SAR)"
            )
        else:
            evidence.append(
                f"CHECK 3c (Texture/speckle): High local texture observed (median Cv={median_cv:.2f}), but capped (0.0 SAR credit) because no prior SAR evidence exists"
            )
    else:
        evidence.append(
            f"CHECK 3c (Texture/speckle): Low/moderate speckle (median Cv={median_cv:.2f}) (0.0 credit)"
        )

    # =========================================================================
    # DECISION LOGIC & CALIBRATION
    # =========================================================================
    total_evidence = optical_score + sar_score
    epsilon = 1e-9

    if total_evidence > 0:
        optical_norm = float(optical_score / (total_evidence + epsilon))
        sar_norm = float(sar_score / (total_evidence + epsilon))
    else:
        optical_norm = 0.0
        sar_norm = 0.0

    margin = abs(optical_norm - sar_norm)
    winning_norm = max(optical_norm, sar_norm)
    winning_modality = "optical" if optical_norm > sar_norm else "sar"

    # Floor check: if total_evidence < 0.40, return 'unknown' directly with confidence 0.0
    if total_evidence < 0.40:
        evidence.append(
            f"DECISION: 'unknown' - Insufficient total evidence (total_evidence={total_evidence:.2f} < floor 0.40; raw optical={optical_score:.2f}, raw sar={sar_score:.2f}; optical_norm={optical_norm:.3f}, sar_norm={sar_norm:.3f}; confidence=0.0)."
        )
        return {
            "modality": "unknown",
            "confidence": 0.0,
            "evidence": evidence
        }

    # Decide optical/sar only if winning_norm > 0.65 AND margin > 0.30
    if winning_norm > 0.65 and margin > 0.30:
        class_max = OPTICAL_MAX_EVIDENCE if winning_modality == "optical" else SAR_MAX_EVIDENCE
        evidence_scale = min(1.0, total_evidence / class_max)
        confidence = round(winning_norm * evidence_scale, 3)

        evidence.append(
            f"DECISION: '{winning_modality}' - Passed all criteria (total_evidence={total_evidence:.2f} >= 0.40, "
            f"winning_norm={winning_norm:.3f} > 0.65, margin={margin:.3f} > 0.30). "
            f"Confidence breakdown: winning_norm={winning_norm:.3f}, class_max={class_max:.2f}, "
            f"evidence_scale={evidence_scale:.3f} -> final confidence={confidence:.3f}."
        )
        return {
            "modality": winning_modality,
            "confidence": confidence,
            "evidence": evidence
        }
    else:
        evidence.append(
            f"DECISION: 'unknown' - Conflicting scores or insufficient margin (total_evidence={total_evidence:.2f} >= 0.40, "
            f"optical_norm={optical_norm:.3f}, sar_norm={sar_norm:.3f}, margin={margin:.3f} <= 0.30 or winning_norm <= 0.65; confidence=0.0)."
        )
        return {
            "modality": "unknown",
            "confidence": 0.0,
            "evidence": evidence
        }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python detect_modality.py <image_path>")
        sys.exit(1)

    target_path = sys.argv[1]
    res = detect_modality(target_path)
    print(json.dumps(res, indent=2))
