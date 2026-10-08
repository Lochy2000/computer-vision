from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(frame_count, sample_count)
    if count == 1:
        return [0]
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def validate_yolo_label(path: Path, class_count: int = 1) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return ["label file is missing"]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 values")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"line {line_number}: values are not numeric")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"line {line_number}: class id {class_id} is out of range")
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"line {line_number}: coordinates must be between 0 and 1")
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"line {line_number}: width and height must be positive")
    return errors


def extract_candidates(
    source: Path,
    output_dir: Path,
    manifest_path: Path,
    sample_count: int,
    source_id: str,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = evenly_spaced_indices(frame_count, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame_index}")
            filename = f"{source_id}-frame-{frame_index:06d}.jpg"
            image_path = output_dir / filename
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write image: {image_path}")
            height, width = frame.shape[:2]
            records.append(
                {
                    "id": image_path.stem,
                    "image": image_path.as_posix(),
                    "image_sha256": sha256_file(image_path),
                    "source_frame_index": frame_index,
                    "timestamp_seconds": frame_index / fps if fps > 0 else None,
                    "width": width,
                    "height": height,
                    "modality": "rgb",
                    "split": "unassigned",
                    "scenario": "close_drone_meadow",
                    "annotation_status": "pending",
                    "label": None,
                }
            )
    finally:
        capture.release()

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "id": source_id,
            "path": source.as_posix(),
            "sha256": sha256_file(source),
            "frame_count": frame_count,
            "fps": fps,
        },
        "classes": [{"id": 0, "name": "drone"}],
        "selection": {"method": "evenly_spaced", "requested": sample_count},
        "images": records,
        "ready_for_accuracy_evaluation": False,
        "blocking_reason": "Human-reviewed bounding-box annotations are not complete.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_manifest(manifest_path: Path) -> tuple[int, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    ready = 0
    for record in manifest.get("images", []):
        image_path = Path(record["image"])
        if not image_path.is_file():
            problems.append(f"{record['id']}: image is missing")
            continue
        if sha256_file(image_path) != record.get("image_sha256"):
            problems.append(f"{record['id']}: image hash has changed")
        if record.get("annotation_status") == "complete":
            label_value = record.get("label")
            if not label_value:
                problems.append(f"{record['id']}: complete annotation has no label path")
                continue
            label_errors = validate_yolo_label(Path(label_value), len(manifest["classes"]))
            problems.extend(f"{record['id']}: {error}" for error in label_errors)
            if not label_errors:
                ready += 1
    return ready, problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and validate a locked evaluation set")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="Extract deterministic annotation candidates")
    extract.add_argument("--source", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--samples", type=int, default=12)
    extract.add_argument("--source-id", required=True)
    validate = subparsers.add_parser("validate", help="Validate images, hashes, and YOLO labels")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        manifest = extract_candidates(
            args.source, args.output_dir, args.manifest, args.samples, args.source_id
        )
        print(f"Extracted {len(manifest['images'])} candidates to {args.output_dir}")
        print("Status: pending human annotation; not ready for accuracy evaluation")
        return
    ready, problems = validate_manifest(args.manifest)
    print(f"Validated annotations ready: {ready}")
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
