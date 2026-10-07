from __future__ import annotations

import argparse
from collections.abc import Sequence

from drone_detection.detectors.ultralytics import UltralyticsDetector
from drone_detection.output import JsonlWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a detector and write model-independent JSONL results."
    )
    parser.add_argument("--source", required=True, help="Video path, stream URL, or camera ID")
    parser.add_argument("--weights", required=True, help="Local Ultralytics .pt weights")
    parser.add_argument("--output", default="runs/detections.jsonl")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device", default=None, help="For example: cpu, 0, or cuda:0")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional model class names to retain, e.g. drone bird aircraft",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from drone_detection.pipeline import process_source

    detector = UltralyticsDetector(
        weights=args.weights,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        allowed_classes=set(args.classes) if args.classes else None,
    )

    frames = 0
    total_inference_ms = 0.0
    with JsonlWriter(args.output) as writer:
        for result in process_source(args.source, detector, args.max_frames):
            writer.write(result)
            frames += 1
            total_inference_ms += result.inference_ms

    average_ms = total_inference_ms / frames if frames else 0.0
    processing_fps = 1000.0 / average_ms if average_ms else 0.0
    print(
        f"Processed {frames} frames; detector mean {average_ms:.2f} ms/frame "
        f"({processing_fps:.2f} FPS); results: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

