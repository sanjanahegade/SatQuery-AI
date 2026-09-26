"""Reusable Optical VQA engine: load GeoChat-7B once, answer many questions."""

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

from prompt_utils import normalize_prompt, extract_comparison_entities
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

from tools.area_comparison import compare_visible_area


DEFAULT_MODEL_PATH = r"D:\SatQueryAI\models\geochat-7B"


def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0

    return torch.cuda.memory_allocated() / (1024 ** 3)


def print_gpu_mem(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        print(
            f"{label}: "
            f"allocated={allocated:.2f} GB, "
            f"reserved={reserved:.2f} GB"
        )


def load_rgb_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"Unsupported or corrupt image: {image_path}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Could not read image: {image_path} ({exc})"
        ) from exc


def is_location_question(question):
    text = (question or "").lower()

    location_terms = (
        "where is",
        "where are",
        "what location",
        "which location",
        "which city",
        "what city",
        "where was",
        "where does",
        "identify the location",
        "location of",
        "name of this place",
    )

    return any(term in text for term in location_terms)


def estimate_answer_confidence(question, answer, visible_text=None):
    """
    Application-level support estimate.
    This is NOT calibrated model probability.
    """

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
        "not sure",
        "unclear",
        "uncertain",
        "cannot determine",
        "can't determine",
        "not enough evidence",
        "insufficient evidence",
        "difficult to determine",
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

    inference_terms = (
        "likely",
        "appears",
        "suggests",
        "may be",
        "possibly",
    )

    if any(term in lower_text for term in inference_terms):
        score -= 0.03
        reasons.append("answer contains qualified inference")

    if any(
        phrase in lower_text
        for phrase in (
            "visible",
            "shown",
            "seen",
            "image",
            "area",
            "building",
            "road",
            "water",
            "vegetation",
        )
    ):
        score += 0.05
        reasons.append("answer cites observable image details")

    score = max(0.0, min(0.95, score))

    return round(score, 2), "; ".join(reasons)


class OpticalVQA:

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path

        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None

        self.load_model()

    def load_model(self):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Optical VQA requires a CUDA GPU."
            )

        disable_torch_init()

        model_name = get_model_name_from_path(self.model_path)

        print("Loading GeoChat...")

        try:
            tokenizer, model, image_processor, context_len = (
                load_pretrained_model(
                    self.model_path,
                    None,
                    model_name,
                    load_4bit=True,
                    device="cuda",
                )
            )

        except Exception as exc:
            raise RuntimeError(
                f"model loading failed: {exc}"
            ) from exc

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        print("GeoChat loaded successfully.")

    def _comparison_answer(self, image_path, question):
        """
        Handle supported visible-area comparison questions separately
        from GeoChat.

        Returns None when the comparison cannot be handled by the
        dedicated visual-area module.
        """

        entities = extract_comparison_entities(question)

        if not entities:
            return None

        entity_a, entity_b = entities

        print(
            f"Detected area comparison: "
            f"{entity_a} vs {entity_b}"
        )

        try:
            result = compare_visible_area(
                image_path,
                entity_a,
                entity_b,
            )
        except Exception as exc:
            print(
                f"Area comparison failed; "
                f"falling back to GeoChat: {exc}"
            )
            return None

        if result is None:
            print(
                "Area comparison does not support these entities; "
                "falling back to GeoChat."
            )
            return None

        winner = result["winner"]

        if winner == "approximately equal":
            answer = (
                f"{entity_a} and {entity_b} occupy "
                f"approximately equal visible area."
            )
        else:
            answer = (
                f"{winner} occupies more visible area."
            )

        return {
            "answer": answer,
            "confidence": result["confidence"],
            "confidence_explanation": (
                "Answer based on a pixel-based visual area estimate."
            ),
            "confidence_type": "visual_area_estimate",
            "visible_text": [],
            "evidence": {
                "method": result["method"],
                "entity_a": result["entity_a"],
                "entity_b": result["entity_b"],
                "entity_a_visible_area_percent": result[
                    "area_a_percent"
                ],
                "entity_b_visible_area_percent": result[
                    "area_b_percent"
                ],
            },
        }

    def answer(self, image_path, question):

        if (
            self.model is None
            or self.tokenizer is None
            or self.image_processor is None
        ):
            raise RuntimeError(
                "Model is not loaded. Call load_model() first."
            )

        image = load_rgb_image(image_path)

        original_question = question

        # --------------------------------------------------
        # DEDICATED AREA COMPARISON
        # --------------------------------------------------
        comparison_result = self._comparison_answer(
            image_path,
            original_question,
        )

        if comparison_result is not None:
            print(
                f"Area comparison answer: "
                f"{comparison_result['answer']}"
            )

            return comparison_result

        # --------------------------------------------------
        # OPTIONAL OCR FOR LOCATION QUESTIONS
        # --------------------------------------------------
        visible_text_evidence = []

        if is_location_question(original_question):

            try:
                ocr_result = extract_visible_text(image_path)

                visible_text_evidence = ocr_result.get(
                    "text",
                    []
                )

            except Exception as exc:
                print(
                    f"OCR evidence unavailable: {exc}"
                )
                visible_text_evidence = []

            if visible_text_evidence:

                question = (
                    f"{original_question}\n\n"
                    "VISIBLE TEXT FROM IMAGE:\n"
                    + "\n".join(
                        f'- "{text}"'
                        for text in visible_text_evidence
                    )
                    + "\n"
                    "Use this visible text as evidence. "
                    "Do not invent location information."
                )

        # --------------------------------------------------
        # PROMPT NORMALIZATION
        # --------------------------------------------------

        question = normalize_prompt(question)

        print(
            f"Original question: {original_question}"
        )

        print(
            f"Normalized question: {question}"
        )

        # --------------------------------------------------
        # GEochat PROMPT
        # --------------------------------------------------

        qs = question

        if self.model.config.mm_use_im_start_end:

            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )

        else:

            qs = (
                DEFAULT_IMAGE_TOKEN
                + "\n"
                + qs
            )

        conv = conv_templates["llava_v1"].copy()

        conv.append_message(
            conv.roles[0],
            qs
        )

        conv.append_message(
            conv.roles[1],
            None
        )

        prompt = conv.get_prompt()

        # --------------------------------------------------
        # TOKENIZE
        # --------------------------------------------------

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).cuda()

        # --------------------------------------------------
        # IMAGE PREPROCESSING
        # --------------------------------------------------

        image_tensor = self.image_processor.preprocess(
            [image],
            crop_size={
                "height": 504,
                "width": 504,
            },
            size={
                "shortest_edge": 504,
            },
            return_tensors="pt",
        )["pixel_values"]

        stop_str = (
            conv.sep
            if conv.sep_style != SeparatorStyle.TWO
            else conv.sep2
        )

        # --------------------------------------------------
        # INFERENCE
        # --------------------------------------------------

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

            raise RuntimeError(
                "CUDA out of memory during inference"
            ) from exc

        except RuntimeError as exc:

            if "out of memory" in str(exc).lower():

                raise RuntimeError(
                    "CUDA out of memory during inference"
                ) from exc

            raise RuntimeError(
                f"inference failed: {exc}"
            ) from exc

        # --------------------------------------------------
        # DECODE
        # --------------------------------------------------

        input_token_len = input_ids.shape[1]

        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=True,
        )

        answer = outputs[0].strip()

        if answer.endswith(stop_str):

            answer = answer[
                :-len(stop_str)
            ]

        answer = answer.strip()

        # --------------------------------------------------
        # LIGHTWEIGHT CLEANUP
        # --------------------------------------------------

        answer = answer.replace(
            "The image appears to be a satellite view of a ",
            "It looks like a ",
        )

        answer = answer.replace(
            "The image appears to be a satellite view of ",
            "It looks like ",
        )

        answer = answer.replace(
            "The presence of ",
            "The visible ",
        )

        # Remove accidental apology openings.

        answer = re.sub(
            r"^(?:I\s+)?"
            r"(?:apologize|am sorry|sorry)"
            r"[^.!?]*[.!?]\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        # --------------------------------------------------
        # TRIM TRAILING FRAGMENTS
        # --------------------------------------------------

        if (
            not re.search(r"[.!?]$", answer)
            and len(answer) > 10
        ):

            last_punc = max(
                answer.rfind(". "),
                answer.rfind("! "),
                answer.rfind("? "),
            )

            if last_punc > 0:

                answer = answer[
                    :last_punc + 1
                ]

            elif answer.rfind(".") > 0:

                answer = answer[
                    :answer.rfind(".") + 1
                ]

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        confidence, confidence_explanation = (
            estimate_answer_confidence(
                original_question,
                answer,
                visible_text_evidence,
            )
        )

        print(
            f"Answer: {answer}"
        )

        print(
            "Application confidence estimate: "
            f"{confidence:.2f} "
            f"({confidence_explanation})"
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_explanation": (
                confidence_explanation
            ),
            "confidence_type": (
                "application_support_estimate"
            ),
            "visible_text": (
                visible_text_evidence or []
            ),
        }