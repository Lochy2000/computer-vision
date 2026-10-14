from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2


V31_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = V31_ROOT.parent
sys.path.insert(0, str(V31_ROOT / "src"))
sys.path.insert(0, str(V3_ROOT / "src"))

from counter_uas.core import Frame  # noqa: E402
from counter_uas.detectors.ultralytics import UltralyticsDetector as CandidateDetector  # noqa: E402
from counter_uas.verification import compare_detections  # noqa: E402
from drone_detection.detectors.ultralytics import UltralyticsDetector as LegacyDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen v3 and v3.1 YOLO adapter outputs."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frame-indices",
        default="0,30,60,90,120",
        help="Comma-separated zero-based video frame indices.",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requested_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("frame indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise ValueError("at least one non-negative frame index is required")
    if len(indices) != len(set(indices)):
        raise ValueError("frame indices must not contain duplicates")
    return tuple(sorted(indices))


def read_frames(source: Path, indices: tuple[int, ...]) -> dict[int, Any]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    frames: dict[int, Any] = {}
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index} from {source}")
            frames[index] = image
    finally:
        capture.release()
    return frames


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    weights = args.weights.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    indices = requested_indices(args.frame_indices)
    frames = read_frames(source, indices)
    allowed_classes = {"UAV", "Bird"}
    common = {
        "weights": weights,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "device": args.device,
        "allowed_classes": allowed_classes,
    }
    print(f"Loaded {len(frames)} comparison frame(s).", flush=True)
    print("Loading frozen v3 detector...", flush=True)
    legacy = LegacyDetector(**common)
    legacy_outputs: dict[int, tuple[Any, ...]] = {}
    for index, image in frames.items():
        print(f"v3 inference: frame {index}", flush=True)
        legacy_outputs[index] = tuple(legacy.predict(image))
    del legacy
    gc.collect()

    print("Loading v3.1 detector...", flush=True)
    candidate = CandidateDetector(**common)
    candidate_outputs: dict[int, tuple[Any, ...]] = {}
    for index, image in frames.items():
        height, width = image.shape[:2]
        frame = Frame(
            frame_id=f"compatibility:{index}",
            sensor_id="recorded-video",
            sequence_number=index,
            captured_at=datetime.fromtimestamp(index, tz=timezone.utc),
            width=width,
            height=height,
        )
        print(f"v3.1 inference: frame {index}", flush=True)
        candidate_outputs[index] = candidate.detect(image, frame)
    del candidate
    gc.collect()

    frame_results: list[dict[str, Any]] = []
    for index, image in frames.items():
        height, width = image.shape[:2]
        legacy_detections = legacy_outputs[index]
        candidate_detections = candidate_outputs[index]
        differences = compare_detections(
            legacy_detections,
            candidate_detections,
            absolute_tolerance=args.tolerance,
        )
        frame_results.append(
            {
                "frame_index": index,
                "width": width,
                "height": height,
                "legacy_detection_count": len(legacy_detections),
                "candidate_detection_count": len(candidate_detections),
                "equivalent": not differences,
                "differences": [difference.to_dict() for difference in differences],
            }
        )

    passed = all(item["equivalent"] for item in frame_results)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "v3 to v3.1 detector-adapter compatibility checkpoint",
        "result": "pass" if passed else "fail",
        "comparison_policy": {
            "ordered_outputs": True,
            "absolute_tolerance": args.tolerance,
            "compared_fields": [
                "count",
                "class_id",
                "class_name",
                "confidence",
                "bbox.x1",
                "bbox.y1",
                "bbox.x2",
                "bbox.y2",
            ],
            "excluded": ["adapter-specific model metadata", "inference timing"],
        },
        "inputs": {
            "source": source.name,
            "source_sha256": sha256(source),
            "weights": weights.name,
            "weights_sha256": sha256(weights),
            "frame_indices": list(indices),
            "image_size": args.image_size,
            "confidence": args.confidence,
            "device": args.device,
            "allowed_classes": sorted(allowed_classes),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ultralytics": version("ultralytics"),
            "opencv": cv2.__version__,
        },
        "summary": {
            "frames_compared": len(frame_results),
            "frames_equivalent": sum(item["equivalent"] for item in frame_results),
            "legacy_detections": sum(
                item["legacy_detection_count"] for item in frame_results
            ),
            "candidate_detections": sum(
                item["candidate_detection_count"] for item in frame_results
            ),
        },
        "frames": frame_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Compatibility result: {report['result'].upper()}")
    print(f"Report: {args.output.resolve()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

