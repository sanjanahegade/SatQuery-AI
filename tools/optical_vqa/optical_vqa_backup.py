"""Reusable Optical VQA engine: load GeoChat-7B once, answer many questions.

Inference path matches tools/test_optical_vqa.py and
geochat/eval/batch_geochat_vqa.py.
"""
import os
import re
import sys

from PIL import Image, UnidentifiedImageError
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prompt_utils import normalize_prompt
from ocr_evidence import extract_visible_text
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

DEFAULT_MODEL_PATH = r"D:\satQai\models\geochat-7B"


def gpu_mem_gb():
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved


def print_gpu_mem(label):
    mem = gpu_mem_gb()
    if mem is None:
        print(f"GPU memory {label}: CUDA not available")
        return
    allocated, reserved = mem
    print(f"GPU memory {label}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


def load_rgb_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        image = Image.open(image_path)
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupt image: {image_path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read image: {image_path} ({exc})") from exc
    return image.convert("RGB")


def is_location_question(question):
    lower_question = (question or "").lower()
    return any(
        phrase in lower_question
        for phrase in (
            "where is",
            "which area",
            "what area",
            "location",
            "city",
            "place",
            "landmark",
            "map label",
            "is this ",
        )
    )


def estimate_answer_confidence(question, answer, visible_text=None):
    """Estimate support strength from the answer text; this is not model calibration."""
    text = (answer or "").strip()
    lower_text = text.lower()
    lower_question = (question or "").lower()
    score = 0.5
    reasons = []

    uncertainty = re.findall(
        r"\b(?:cannot|can't|unable|unclear|uncertain|difficult to determine|not enough evidence|ambiguous|可能|possibly|perhaps)\b",
        lower_text,
    )
    speculation = re.findall(r"\b(?:likely|might|may be|could be|appears to be|suggests)\b", lower_text)
    evidence_terms = re.findall(
        r"\b(?:visible|shown|seen|readable|label|labels|text|sign|building|road|water|vegetation|ship|harbor)\b",
        lower_text,
    )
    location_question = any(word in lower_question for word in ("where", "location", "city", "place", "which area"))
    text_evidence = any(word in lower_text for word in ("label", "text", "sign", "gandhi", "cubbon", "bengaluru", "bangalore"))
    visible_text = visible_text or []
    visible_text_string = " ".join(visible_text).lower()
    visible_location_evidence = any(
        place in visible_text_string
        for place in ("gandhi nagar", "cubbon park", "bengaluru", "bangalore")
    )

    if evidence_terms:
        score += 0.12
        reasons.append("answer cites observable image details")
    if text_evidence:
        score += 0.12
        reasons.append("answer cites visible text or map-label evidence")
    if location_question and visible_location_evidence:
        score += 0.18
        reasons.append("OCR found recognizable location text")
    if uncertainty:
        score -= min(0.24, 0.08 * len(uncertainty))
        reasons.append("answer explicitly reports uncertainty")
    if speculation:
        score -= min(0.18, 0.06 * len(speculation))
        reasons.append("answer contains qualified inference")
    if location_question and not text_evidence:
        score = min(score, 0.55)
        reasons.append("location answer has no visible-text evidence")
    if len(text.split()) < 5:
        score -= 0.08
        reasons.append("answer provides little supporting detail")

    score = max(0.05, min(0.95, round(score, 2)))
    if not reasons:
        reasons.append("estimate is based on answer specificity only")
    level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    explanation = f"Application-level {level} support estimate, not calibrated model confidence: " + "; ".join(reasons) + "."
    return score, explanation


class OpticalVQA:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = os.path.expanduser(model_path)
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

        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                self.model_path,
                None,
                model_name,
                load_4bit=True,
                device="cuda",
            )
        except Exception as exc:
            raise RuntimeError(f"model loading failed: {exc}") from exc

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

    def answer(self, image_path, question):
        if self.model is None or self.tokenizer is None or self.image_processor is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        image = load_rgb_image(image_path)

        original_question = question
        visible_text_evidence = []
        if is_location_question(original_question):
            ocr_result = extract_visible_text(image_path)
            visible_text_evidence = ocr_result.get("text", [])
            if visible_text_evidence:
                question = (
                    f"{original_question}\n\nVISIBLE TEXT FROM IMAGE:\n"
                    + "\n".join(f'- "{text}"' for text in visible_text_evidence)
                    + "\nUse this visible text as evidence. Do not invent location information."
                )
        question = normalize_prompt(question)
        print(f"Original question: {original_question}")
        print(f"Normalized question: {question}")

        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # Official GeoChat VQA preprocess (geochat/eval/batch_geochat_vqa.py): 504x504 CLIP.
        image_tensor = self.image_processor.preprocess(
            [image],
            crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504},
            return_tensors="pt",
        )["pixel_values"]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("CUDA out of memory during inference") from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CUDA out of memory during inference") from exc
            raise RuntimeError(f"inference failed: {exc}") from exc
"""Reusable Optical VQA engine: load GeoChat-7B once, answer many questions.

Inference path matches tools/test_optical_vqa.py and
geochat/eval/batch_geochat_vqa.py.
"""
import os
import sys

from PIL import Image, UnidentifiedImageError
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prompt_utils import normalize_prompt
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

DEFAULT_MODEL_PATH = r"D:\satQai\models\geochat-7B"


def gpu_mem_gb():
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved


def print_gpu_mem(label):
    mem = gpu_mem_gb()
    if mem is None:
        print(f"GPU memory {label}: CUDA not available")
        return
    allocated, reserved = mem
    print(f"GPU memory {label}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


def load_rgb_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        image = Image.open(image_path)
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupt image: {image_path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read image: {image_path} ({exc})") from exc
    return image.convert("RGB")


class OpticalVQA:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = os.path.expanduser(model_path)
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

        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                self.model_path,
                None,
                model_name,
                load_4bit=True,
                device="cuda",
            )
        except Exception as exc:
            raise RuntimeError(f"model loading failed: {exc}") from exc

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

    def answer(self, image_path, question):
        if self.model is None or self.tokenizer is None or self.image_processor is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        image = load_rgb_image(image_path)

        original_question = question
        visible_text_evidence = []
        if is_location_question(original_question):
            ocr_result = extract_visible_text(image_path)
            visible_text_evidence = ocr_result.get("text", [])
            if visible_text_evidence:
                question = (
                    f"{original_question}\n\nVISIBLE TEXT FROM IMAGE:\n"
                    + "\n".join(f'- \"{text}\"' for text in visible_text_evidence)
                    + "\nUse this visible text as evidence. Do not invent location information."
                )
        question = normalize_prompt(question)
        print(f"Original question: {original_question}")
        print(f"Normalized question: {question}")

        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # Official GeoChat VQA preprocess (geochat/eval/batch_geochat_vqa.py): 504x504 CLIP.
        image_tensor = self.image_processor.preprocess(
            [image],
            crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504},
            return_tensors="pt",
        )["pixel_values"]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("CUDA out of memory during inference") from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CUDA out of memory during inference") from exc
            raise RuntimeError(f"inference failed: {exc}") from exc

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )
        
        answer = outputs[0].strip()
        if answer.endswith(stop_str):
            answer = answer[: -len(stop_str)]
        answer = answer.strip()
        
        # Lightweight cleanup for repetitive GeoChat phrases
        answer = answer.replace("The image appears to be a satellite view of a ", "It looks like a ")
        answer = answer.replace("The image appears to be a satellite view of ", "It looks like ")
        answer = answer.replace("The presence of ", "The visible ")

        # Keep challenge responses evidence-focused instead of apologetic.
        answer = re.sub(
            r"^(?:I\s+)?(?:apologize|am sorry|sorry)[^.!?]*[.!?]\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        # Trim accidental trailing sentence fragments
        if not re.search(r'[.!?]$', answer) and len(answer) > 10:
            last_punc = max(answer.rfind('. '), answer.rfind('! '), answer.rfind('? '))
            if last_punc > 0:
                answer = answer[:last_punc+1]
            elif answer.rfind('.') > 0:
                answer = answer[:answer.rfind('.')+1]
                
        confidence, confidence_explanation = estimate_answer_confidence(
            original_question,
            answer,
            visible_text_evidence,
        )
        print(f"Answer: {answer}")
        print(f"Application confidence estimate: {confidence:.2f} ({confidence_explanation})")
        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_explanation": confidence_explanation,
            "confidence_type": "application_support_estimate",
            "visible_text": visible_text_evidence or [],
        }
