from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from counter_uas.evaluation import load_observation_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render frozen observations over their source video.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence = load_observation_sequence(args.fixture)
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    first = sequence.frames[0].frame
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.width, first.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")
    try:
        for batch in sequence.replay():
            capture.set(cv2.CAP_PROP_POS_FRAMES, batch.frame.sequence_number)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {batch.frame.sequence_number}")
            for observation in batch.observations:
                detection = observation.detection
                box = detection.bbox
                colour = (0, 0, 255) if detection.confidence >= 0.25 else (0, 255, 255)
                cv2.rectangle(
                    image,
                    (round(box.x1), round(box.y1)),
                    (round(box.x2), round(box.y2)),
                    colour,
                    3,
                )
                cv2.putText(
                    image,
                    f"{detection.class_name} {detection.confidence:.2f}",
                    (round(box.x1), max(25, round(box.y1) - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                image,
                f"source frame {batch.frame.sequence_number} | frozen observations {len(batch.observations)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()
        capture.release()
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
