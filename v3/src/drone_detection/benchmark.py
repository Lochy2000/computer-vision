from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from drone_detection.detectors.ultralytics import UltralyticsDetector
from drone_detection.pipeline import process_source


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(path: str | Path) -> dict[str, str]:
    artifact = Path(path)
    if artifact.is_file():
        return {artifact.name: sha256_file(artifact)}
    return {
        str(item.relative_to(artifact)): sha256_file(item)
        for item in sorted(artifact.rglob("*"))
        if item.is_file()
    }


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def summarise(times_ms: list[float], detection_counts: list[int]) -> dict[str, Any]:
    steady = times_ms[1:] if len(times_ms) > 1 else []
    steady_mean = statistics.fmean(steady) if steady else None
    return {
        "frames": len(times_ms),
        "frames_with_detections": sum(count > 0 for count in detection_counts),
        "total_detections": sum(detection_counts),
        "first_frame_ms": times_ms[0] if times_ms else None,
        "mean_inference_ms_including_startup": statistics.fmean(times_ms) if times_ms else None,
        "steady_mean_inference_ms": steady_mean,
        "steady_median_inference_ms": statistics.median(steady) if steady else None,
        "steady_fps": 1000.0 / steady_mean if steady_mean else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a portable, auditable detector benchmark record.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source)
    weights = Path(args.weights)
    if not source.is_file():
        raise FileNotFoundError(f"Benchmark source not found: {source}")

    detector = UltralyticsDetector(
        weights=weights,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        allowed_classes=set(args.classes) if args.classes else None,
    )

    times_ms: list[float] = []
    detection_counts: list[int] = []
    wall_started = perf_counter()
    for result in process_source(source, detector, max_frames=args.max_frames):
        times_ms.append(result.inference_ms)
        detection_counts.append(len(result.detections))
    wall_seconds = perf_counter() - wall_started

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": args.model_id,
            "path": str(weights),
            "artifact_sha256": artifact_hashes(weights),
        },
        "test": {
            "id": args.test_id,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "max_frames": args.max_frames,
        },
        "inference": {
            "device": args.device,
            "image_size": args.image_size,
            "confidence": args.confidence,
            "classes": args.classes,
            "wall_seconds": wall_seconds,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "packages": {
                name: package_version(name)
                for name in ("torch", "ultralytics", "openvino", "opencv-python", "numpy")
            },
        },
        "metrics": summarise(times_ms, detection_counts),
        "limitations": [
            "Detection counts are not accuracy metrics without ground-truth annotations.",
            "Frames after the first are reported separately to reduce cold-start distortion.",
            "Results are specific to the recorded model, media hash, settings, runtime, and device.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Benchmark record: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
