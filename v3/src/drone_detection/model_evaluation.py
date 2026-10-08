from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence


EXPECTED_SPLIT_COUNTS = {"train": 5367, "val": 876, "test": 875}
EXPECTED_CANONICAL_HASHES = {
    "train": "2e6100e2c3f9cd5019f45bd90b4090cc7bc5ee270ae97d020c21245c86832921",
    "val": "c6215299af7eedc0baf70e394b64ce18519f0937be267a433568a1008465927a",
    "test": "e931bf035b9e091b019ad3c0576fe36dd05e8d2be774939ce4e181c1a3445d92",
}
EXPECTED_CLASSES = {0: "UAV", 1: "Bird"}
CONFIRMATION = "evaluate-locked-test-once"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_split(path: Path) -> list[str]:
    entries = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate entries found in {path}")
    return entries


def canonical_hash(entries: list[str]) -> str:
    content = "\n".join(sorted(entries)) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_locked_splits(dataset_root: Path, split_root: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    splits: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    all_names: set[str] = set()
    for split in ("train", "val", "test"):
        entries = read_split(split_root / f"{split}.txt")
        if len(entries) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(f"{split} contains {len(entries)} entries; expected {EXPECTED_SPLIT_COUNTS[split]}")
        digest = canonical_hash(entries)
        if digest != EXPECTED_CANONICAL_HASHES[split]:
            raise ValueError(f"{split} membership hash {digest} does not match the locked scene-v1 split")
        overlap = all_names.intersection(entries)
        if overlap:
            raise ValueError(f"{split} overlaps an earlier split by {len(overlap)} filenames")
        all_names.update(entries)
        for entry in entries:
            image = dataset_root / entry
            label = dataset_root / "labels" / f"{Path(entry).stem}.txt"
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"Missing locked-test image or label for {entry}")
        splits[split] = entries
        hashes[split] = digest
    return splits, hashes


def write_runtime_dataset(dataset_root: Path, splits: dict[str, list[str]], artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_files: dict[str, Path] = {}
    for split, entries in splits.items():
        resolved = artifact_dir / f"{split}.resolved.txt"
        paths = [(dataset_root / entry).resolve().as_posix() for entry in entries]
        resolved.write_text("\n".join(paths) + "\n", encoding="utf-8")
        resolved_files[split] = resolved

    yaml_path = artifact_dir / "dataset.generated.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {json.dumps(dataset_root.resolve().as_posix())}",
                f"train: {json.dumps(resolved_files['train'].resolve().as_posix())}",
                f"val: {json.dumps(resolved_files['val'].resolve().as_posix())}",
                f"test: {json.dumps(resolved_files['test'].resolve().as_posix())}",
                "names:",
                "  0: UAV",
                "  1: Bird",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def scalar(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def metrics_from_result(result: Any) -> dict[str, Any]:
    names = {int(key): str(value) for key, value in dict(result.names).items()}
    precision = list(result.box.p)
    recall = list(result.box.r)
    ap50 = list(result.box.ap50)
    ap = list(result.box.ap)
    per_class = {
        names[index]: {
            "precision": scalar(precision[index]),
            "recall": scalar(recall[index]),
            "map50": scalar(ap50[index]),
            "map50_95": scalar(ap[index]),
        }
        for index in sorted(names)
        if index < min(len(precision), len(recall), len(ap50), len(ap))
    }
    return {
        "overall": {
            "precision": scalar(result.box.mp),
            "recall": scalar(result.box.mr),
            "map50": scalar(result.box.map50),
            "map50_95": scalar(result.box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {str(key): scalar(value) for key, value in result.speed.items()},
        "ultralytics_results": {str(key): scalar(value) for key, value in result.results_dict.items()},
    }


def training_artifact_hashes(weights: Path) -> dict[str, str]:
    run_dir = weights.parent.parent
    names = ("args.yaml", "results.csv", "preflight.json", "dataset.generated.yaml")
    return {name: sha256_file(run_dir / name) for name in names if (run_dir / name).is_file()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one trained checkpoint on the locked UB-SOD scene-v1 test split")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--confirm", required=True, help=f"Must be exactly: {CONFIRMATION}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.confirm != CONFIRMATION:
        raise ValueError(f"Locked test not evaluated: --confirm must be exactly {CONFIRMATION!r}")
    if args.output.exists():
        raise FileExistsError(f"Evaluation record already exists and will not be overwritten: {args.output}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    dataset_root = args.dataset_root.resolve()
    split_root = args.split_root.resolve()
    splits, split_hashes = validate_locked_splits(dataset_root, split_root)
    dataset_yaml = write_runtime_dataset(dataset_root, splits, args.artifacts_dir.resolve())

    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    validation = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        plots=True,
        project=str(args.artifacts_dir.resolve().parent),
        name=args.artifacts_dir.name,
        exist_ok=True,
    )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_policy": {
            "split": "test",
            "purpose": "single locked-test evaluation after model selection",
            "threshold_tuning_permitted": False,
        },
        "model": {
            "id": args.model_id,
            "checkpoint": str(args.weights.resolve()),
            "checkpoint_sha256": sha256_file(args.weights),
            "training_artifact_sha256": training_artifact_hashes(args.weights),
        },
        "dataset": {
            "id": "ub-sod-scene-v1",
            "root": str(dataset_root),
            "classes": EXPECTED_CLASSES,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "canonical_split_sha256": split_hashes,
            "evaluated_images": len(splits["test"]),
        },
        "evaluation": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "artifacts_dir": str(args.artifacts_dir.resolve()),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "packages": {name: package_version(name) for name in ("torch", "ultralytics", "numpy")},
        },
        "metrics": metrics_from_result(validation),
        "limitations": [
            "The split is grouped by inferred visual similarity because authoritative recording/session IDs are unavailable.",
            "UB-SOD contains RGB airport-domain imagery and does not establish thermal performance.",
            "Independent negative videos are still required to measure false drone alerts per hour.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Locked-test evaluation record: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
