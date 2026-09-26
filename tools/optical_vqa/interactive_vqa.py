"""Interactive Optical VQA: load GeoChat once, then ask multiple questions."""
import argparse
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from optical_vqa import OpticalVQA, load_rgb_image


def main():
    parser = argparse.ArgumentParser(description="Interactive GeoChat optical VQA")
    parser.add_argument("image_path", help="Path to one optical satellite image")
    args = parser.parse_args()
    image_path = args.image_path

    try:
        load_rgb_image(image_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Loading GeoChat...")
    try:
        vqa = OpticalVQA()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("Model loaded.")
    print()
    print("Interactive Optical VQA")
    print("Type a question, or 'exit' to quit.")
    print()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Goodbye.")
            return 0

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print()
            print("Goodbye.")
            return 0

        try:
            answer = vqa.answer(image_path, question)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            continue
        except ValueError as exc:
            print(f"ERROR: {exc}")
            continue
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            continue

        print(f"Answer: {answer}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
