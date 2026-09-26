"""Optional visible-text extraction for Optical VQA evidence."""
import os
import re


def extract_visible_text(image_path):
    """Return OCR text and confidence, or a graceful unavailable/error result."""
    result = {
        "text": [],
        "confidence": None,
        "available": False,
        "error": None,
    }
    if not image_path or not os.path.isfile(image_path):
        result["error"] = "image file not found"
        return result

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        result["error"] = "no supported OCR capability is installed"
        return result

    try:
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = []
        confidences = []
        for word, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            cleaned = re.sub(r"\s+", " ", (word or "")).strip()
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                confidence = -1
            if cleaned and confidence >= 0:
                words.append(cleaned)
                confidences.append(confidence)

        result["text"] = words
        result["confidence"] = round(sum(confidences) / len(confidences) / 100, 2) if confidences else None
        result["available"] = True
        return result
    except Exception as exc:
        result["error"] = f"OCR failed: {exc}"
        return result
