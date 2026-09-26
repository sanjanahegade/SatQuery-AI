#!/usr/bin/env python3
"""CLI interface for SatQueryAI Agentic Controller & Router.

Usage:
    python router_cli.py --images <image_path> --query "Describe this satellite image."
    python router_cli.py --images <pre.tif> <post.tif> --query "What changed between these images?"
    python router_cli.py --images <optical.tif> <sar.tif> --query "Use optical and SAR together to identify built-up and water."
"""

import argparse
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from router import run_satquery


def main():
    parser = argparse.ArgumentParser(description="SatQueryAI Router & Controller CLI")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more paths to input satellite images (Optical GeoTIFF/PNG/JPG or SAR GeoTIFF/PNG)",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language question or instruction for SatQueryAI",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock specialist execution for rapid routing verification without model loading",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to write the JSON result to disk",
    )
    args = parser.parse_args()

    result = run_satquery(args.images, args.query, mock_specialist=args.mock)

    formatted_json = json.dumps(result, indent=2)
    print(formatted_json)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(formatted_json)


if __name__ == "__main__":
    main()
