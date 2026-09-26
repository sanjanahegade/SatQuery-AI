"""Minimal Optical VQA smoke test: one image + one question → one GeoChat answer.

Reuses the official inference path from geochat/eval/batch_geochat_vqa.py.
"""
import argparse
import os
import sys

from PIL import Image, UnidentifiedImageError
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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

DEFAULT_MODEL_PATH = r"D:\SatQueryAI\models\geochat-7B"


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


def main():
    parser = argparse.ArgumentParser(description="Minimal GeoChat optical VQA test")
    parser.add_argument("image_path", help="Path to one optical satellite image")
    parser.add_argument("question", help="Natural-language VQA question")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="GeoChat checkpoint directory",
    )
    args = parser.parse_args()

    try:
        image = load_rgb_image(args.image_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            None,
            model_name,
            load_4bit=True,
            device="cuda",
        )
    except Exception as exc:
        print(f"ERROR: model loading failed: {exc}")
        return 1

    qs = args.question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    # Official GeoChat VQA preprocess (geochat/eval/batch_geochat_vqa.py): 504x504 CLIP.
    image_tensor = image_processor.preprocess(
        [image],
        crop_size={"height": 504, "width": 504},
        size={"shortest_edge": 504},
        return_tensors="pt",
    )["pixel_values"]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    print_gpu_mem("before inference")

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=64,
                use_cache=True,
            )
    except torch.cuda.OutOfMemoryError:
        print("ERROR: CUDA out of memory during inference")
        return 1
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("ERROR: CUDA out of memory during inference")
            return 1
        print(f"ERROR: inference failed: {exc}")
        return 1

    print_gpu_mem("after inference")

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    output = outputs[0].strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)]
    output = output.strip()

    print()
    print("ANSWER:")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
