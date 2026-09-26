"""Reusable Optical VQA and Scene Captioning engine using GeoChat-7B in 4-bit mode."""

import os
import re
import sys

# Force offline mode for Hugging Face Hub to prevent remote HTTP latency
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 

from PIL import Image, UnidentifiedImageError
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
GEOCHAT_ROOT = r"D:\satQai\GeoChat"
DEFAULT_MODEL_PATH = r"D:\satQai\models\geochat-7B"

for p in [TOOLS_DIR, GEOCHAT_ROOT, os.path.join(GEOCHAT_ROOT, "tools")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from prompt_utils import normalize_prompt, extract_comparison_entities
except ImportError:
    normalize_prompt = lambda x: x
    extract_comparison_entities = lambda x: None

try:
    from ocr_evidence import extract_visible_text
except ImportError:
    extract_visible_text = lambda x: {"text": [], "available": False}

from geochat.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)

from geochat.conversation import SeparatorStyle, conv_templates
from geochat.mm_utils import get_model_name_from_path, tokenizer_image_token
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init

try:
    from area_comparison import compare_visible_area
except ImportError:
    try:
        from tools.area_comparison import compare_visible_area
    except ImportError:
        compare_visible_area = None


def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)


def print_gpu_mem(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"{label}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


def load_rgb_image(image_path):
    """Load an image as an RGB PIL Image, supporting both standard formats (JPG, PNG)
    and geospatial multi-band GeoTIFFs (e.g. Sentinel-2 uint16 rasters).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = Image.open(image_path)
        img.load()
        return img.convert("RGB")
    except Exception:
        # Fallback using rasterio for GeoTIFF / satellite imagery
        try:
            import rasterio
            import numpy as np

            with rasterio.open(image_path) as src:
                if src.count >= 3:
                    r = src.read(1).astype(np.float32)
                    g = src.read(2).astype(np.float32)
                    b = src.read(3).astype(np.float32)
                    arr = np.stack([r, g, b], axis=-1)
                else:
                    gray = src.read(1).astype(np.float32)
                    arr = np.stack([gray, gray, gray], axis=-1)

                p2 = np.percentile(arr, 2.0)
                p98 = np.percentile(arr, 98.0)
                if p98 > p2:
                    arr = np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

                return Image.fromarray(arr, mode="RGB")
        except Exception as exc:
            raise ValueError(f"Could not read image: {image_path} ({exc})") from exc


def is_location_question(question):
    text = (question or "").lower()
    location_terms = (
        "where is", "where are", "what location", "which location",
        "which city", "what city", "where was", "where does",
        "identify the location", "location of", "name of this place",
    )
    return any(term in text for term in location_terms)


def estimate_answer_confidence(question, answer, visible_text=None):
    text = (answer or "").strip()
    lower_text = text.lower()
    score = 0.50
    reasons = []

    if not text:
        return 0.0, "no answer returned"

    if len(text) >= 20:
        score += 0.05
        reasons.append("answer contains supporting detail")
    else:
        reasons.append("answer provides little supporting detail")

    uncertainty_terms = (
        "not sure", "unclear", "uncertain", "cannot determine",
        "can't determine", "not enough evidence", "insufficient evidence",
    )
    if any(term in lower_text for term in uncertainty_terms):
        score -= 0.15
        reasons.append("answer explicitly reports uncertainty")

    if visible_text:
        score += 0.10
        reasons.append("answer has visible-text or map-label evidence")

    question_lower = (question or "").lower()
    if is_location_question(question_lower) and not visible_text:
        score -= 0.05
        reasons.append("location answer has no visible-text evidence")

    score = max(0.0, min(0.95, score))
    return round(score, 2), "; ".join(reasons)


class OpticalVQA:
    """GeoChat-7B optical VQA and scene captioning engine."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.load_model()

    def load_model(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Optical VQA requires a CUDA GPU.")

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        print(f"Loading GeoChat-7B from {self.model_path} in 4-bit...")

        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                self.model_path,
                None,
                model_name,
                load_4bit=True,
                device="cuda",
            )
        except Exception as exc:
            raise RuntimeError(f"Model loading failed: {exc}") from exc

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        print("GeoChat-7B loaded successfully.")

    def _generate(self, image, prompt_text, max_new_tokens=128):
        """Core generation routine for both VQA and Captioning."""
        if self.model is None or self.tokenizer is None or self.image_processor is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        qs = prompt_text
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).cuda()

        image_tensor = self.image_processor.preprocess(
            [image],
            crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504},
            return_tensors="pt",
        )["pixel_values"]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=True,
        )

        response = outputs[0].strip()
        if response.endswith(stop_str):
            response = response[:-len(stop_str)]
        return response.strip()

    def caption_image(self, image_path, prompt="Describe the land-cover and major objects visible in this image."):
        """Generate a natural-language description/caption of the land-cover and features in an optical image."""
        image = load_rgb_image(image_path)
        caption = self._generate(image, prompt, max_new_tokens=160)

        # Clean up any trailing broken sentences
        if not re.search(r"[.!?]$", caption) and len(caption) > 15:
            last_punc = max(caption.rfind(". "), caption.rfind("! "), caption.rfind("? "))
            if last_punc > 0:
                caption = caption[:last_punc + 1]
            elif caption.rfind(".") > 0:
                caption = caption[:caption.rfind(".") + 1]

        return caption

    def answer(self, image_path, question):
        """Answer a natural-language question about an optical satellite image."""
        image = load_rgb_image(image_path)
        original_question = question

        # Area comparison check
        if compare_visible_area is not None:
            entities = extract_comparison_entities(question)
            if entities:
                try:
                    comp_res = compare_visible_area(image_path, entities[0], entities[1])
                    if comp_res:
                        return {
                            "answer": f"{comp_res['winner']} occupies more visible area." if comp_res['winner'] != "approximately equal" else f"{entities[0]} and {entities[1]} occupy approximately equal visible area.",
                            "confidence": comp_res["confidence"],
                            "confidence_explanation": "Answer based on visual area estimate.",
                            "visible_text": [],
                        }
                except Exception:
                    pass

        # Optional OCR evidence
        visible_text_evidence = []
        if is_location_question(original_question):
            try:
                ocr_result = extract_visible_text(image_path)
                visible_text_evidence = ocr_result.get("text", [])
            except Exception:
                visible_text_evidence = []

        norm_question = normalize_prompt(question)
        if visible_text_evidence:
            norm_question += "\n\nVISIBLE TEXT FROM IMAGE:\n" + "\n".join(f'- "{t}"' for t in visible_text_evidence)

        answer_text = self._generate(image, norm_question, max_new_tokens=128)

        # Standard post-cleaning
        answer_text = answer_text.replace("The image appears to be a satellite view of a ", "It looks like a ")
        answer_text = answer_text.replace("The image appears to be a satellite view of ", "It looks like ")

        confidence, explanation = estimate_answer_confidence(original_question, answer_text, visible_text_evidence)
        return {
            "answer": answer_text,
            "confidence": confidence,
            "confidence_explanation": explanation,
            "visible_text": visible_text_evidence or [],
        }


# Global engine singleton for reuse across queries
_ENGINE_INSTANCE = None

def get_engine(model_path=DEFAULT_MODEL_PATH):
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = OpticalVQA(model_path=model_path)
    return _ENGINE_INSTANCE

def caption_image(image_path, prompt="Describe the land-cover and major objects visible in this image.", engine=None):
    """Module-level captioning function as requested by the specification."""
    if engine is None:
        engine = get_engine()
    return engine.caption_image(image_path, prompt=prompt)

def answer_question(image_path, question, engine=None):
    """Module-level VQA function."""
    if engine is None:
        engine = get_engine()
    return engine.answer(image_path, question)
