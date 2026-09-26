"""Demonstrate OpticalVQA: one image, two questions, GeoChat loaded once."""
import argparse
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from optical_vqa import OpticalVQA, print_gpu_mem

QUESTIONS = [
    "What can you see in this image?",
    "Describe the main land cover and any notable features.",
]


def main():
    parser = argparse.ArgumentParser(description="Persistent GeoChat optical VQA test")
    parser.add_argument("image_path", help="Path to one optical satellite image")
    args = parser.parse_args()

    print("Loading model...")
    try:
        vqa = OpticalVQA()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("Model loaded.")
    print()

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"Question {i}:")
        print(question)
        print_gpu_mem("before inference")
        try:
            answer = vqa.answer(args.image_path, question)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1
        print_gpu_mem("after inference")
        print("Answer:")
        print(answer)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
