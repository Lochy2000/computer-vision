from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from counter_uas.core import Frame  # noqa: E402
from counter_uas.detectors.ultralytics import UltralyticsDetector  # noqa: E402
from counter_uas.evaluation import (  # noqa: E402
    FrameObservations,
    ObservationSequence,
    save_observation_sequence,
)
from counter_uas.evaluation.recorded import sha256_file, summarize_observations  # noqa: E402
from counter_uas.observations import build_observations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a detector once over a consecutive video window and freeze its observations."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sensor-id", default="recorded-camera-01")
    parser.add_argument("--fixture-id", default="pexels-meadow-frames-0060-0120")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("frame range must be non-negative and inclusive")
    if not args.source.is_file():
        raise FileNotFoundError(f"Video not found: {args.source}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video metadata is incomplete")
    if args.end_frame >= total_frames:
        raise ValueError(f"end frame {args.end_frame} exceeds final frame {total_frames - 1}")

    detector = UltralyticsDetector(
        args.weights,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        allowed_classes={"UAV", "Bird"},
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    batches: list[FrameObservations] = []
    started = time.perf_counter()
    clock_origin = datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        for frame_index in range(args.start_frame, args.end_frame + 1):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode source frame {frame_index}")
            frame = Frame(
                frame_id=f"source-frame-{frame_index:06d}",
                sensor_id=args.sensor_id,
                sequence_number=frame_index,
                captured_at=clock_origin + timedelta(seconds=frame_index / fps),
                width=width,
                height=height,
            )
            detections = detector.detect(image, frame)
            batches.append(FrameObservations(frame, build_observations(frame, detections)))
            print(f"frame {frame_index}/{args.end_frame}: {len(detections)} observations", flush=True)
    finally:
        capture.release()

    sequence = ObservationSequence(
        fixture_id=args.fixture_id,
        source_id=args.source.stem,
        sensor_id=args.sensor_id,
        frames=tuple(batches),
    )
    save_observation_sequence(sequence, args.fixture)
    manifest = {
        "checkpoint": "3.2.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": str(args.fixture),
        "fixture_sha256": sha256_file(args.fixture),
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "model_id": detector.model_id,
        "video": {"fps": fps, "width": width, "height": height, "total_frames": total_frames},
        "window": {"start_frame": args.start_frame, "end_frame": args.end_frame, "inclusive": True},
        "inference": {
            "image_size": args.image_size,
            "confidence": args.confidence,
            "device": args.device or "ultralytics-default",
            "allowed_classes": ["UAV", "Bird"],
        },
        "timestamp_basis": (
            "relative video position anchored to Unix epoch; not the real-world capture time"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summarize_observations(sequence),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
