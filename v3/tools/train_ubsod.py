"""Preflight and train a reproducible UB-SOD scene-v1 YOLO baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "UB-SOD-release" / "verified-tar"
DEFAULT_SPLIT_ROOT = REPO_ROOT / "splits" / "ub-sod-scene-v1"
EXPECTED_COUNTS = {"train": 5367, "val": 876, "test": 875}
EXPECTED_CANONICAL_HASHES = {
    "train": "2e6100e2c3f9cd5019f45bd90b4090cc7bc5ee270ae97d020c21245c86832921",
    "val": "c6215299af7eedc0baf70e394b64ce18519f0937be267a433568a1008465927a",
    "test": "e931bf035b9e091b019ad3c0576fe36dd05e8d2be774939ce4e181c1a3445d92",
}
CLASS_NAMES = {0: "UAV", 1: "Bird"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> list[str]:
    entries = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate entries found in {path}")
    return entries


def preflight(dataset_root: Path, split_root: Path) -> dict:
    dataset_root = dataset_root.resolve()
    split_root = split_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    splits: dict[str, list[str]] = {}
    split_hashes: dict[str, str] = {}
    canonical_split_hashes: dict[str, str] = {}
    missing: list[str] = []
    bad_labels: list[str] = []

    for split, expected in EXPECTED_COUNTS.items():
        split_file = split_root / f"{split}.txt"
        entries = read_split(split_file)
        if len(entries) != expected:
            raise ValueError(f"{split_file} has {len(entries)} entries; expected {expected}")
        splits[split] = entries
        split_hashes[split] = sha256(split_file)
        canonical_content = "\n".join(sorted(entries)) + "\n"
        canonical_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()
        canonical_split_hashes[split] = canonical_hash
        if canonical_hash != EXPECTED_CANONICAL_HASHES[split]:
            raise ValueError(
                f"{split_file} membership hash is {canonical_hash}; "
                f"expected audited scene-v1 hash {EXPECTED_CANONICAL_HASHES[split]}"
            )

        for relative_image in entries:
            image = dataset_root / relative_image
            label = dataset_root / "labels" / f"{Path(relative_image).stem}.txt"
            if not image.is_file() or not label.is_file():
                missing.append(relative_image)
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except (IndexError, ValueError):
                    bad_labels.append(f"{label}:{line_number}")
                    continue
                if len(coords) != 4 or class_id not in CLASS_NAMES or not all(0.0 <= value <= 1.0 for value in coords):
                    bad_labels.append(f"{label}:{line_number}")

    overlap = {
        "train_val": sorted(set(splits["train"]) & set(splits["val"])),
        "train_test": sorted(set(splits["train"]) & set(splits["test"])),
        "val_test": sorted(set(splits["val"]) & set(splits["test"])),
    }
    if missing or bad_labels or any(overlap.values()):
        raise ValueError(
            f"Preflight failed: missing={len(missing)}, bad_labels={len(bad_labels)}, "
            f"cross_split_overlap={sum(len(values) for values in overlap.values())}"
        )

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "split_root": str(split_root),
        "classes": CLASS_NAMES,
        "counts": {name: len(entries) for name, entries in splits.items()},
        "split_sha256": split_hashes,
        "canonical_split_sha256": canonical_split_hashes,
        "missing_images_or_labels": 0,
        "invalid_labels": 0,
        "cross_split_filename_overlap": 0,
    }


def write_resolved_split(dataset_root: Path, source: Path, output: Path) -> None:
    entries = read_split(source)
    resolved = [(dataset_root.resolve() / entry).as_posix() for entry in entries]
    output.write_text("\n".join(resolved) + "\n", encoding="utf-8")


def write_training_yaml(dataset_root: Path, run_dir: Path, output: Path) -> None:
    # Absolute paths make resolution unambiguous on Windows and in Colab. The
    # generated file is a run artifact; source split files remain portable.
    lines = [
        f"path: {json.dumps(dataset_root.resolve().as_posix())}",
        f"train: {json.dumps((run_dir / 'train.resolved.txt').as_posix())}",
        f"val: {json.dumps((run_dir / 'val.resolved.txt').as_posix())}",
        "names:",
        "  0: UAV",
        "  1: Bird",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 enables Ultralytics automatic sizing")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the training split to use")
    parser.add_argument("--save-period", type=int, default=5, help="Save an additional checkpoint every N epochs")
    parser.add_argument("--device", default=None, help="For example 0, cpu, or mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "train")
    parser.add_argument("--name", default="ubsod-scene-v1-yolo26n-640")
    parser.add_argument("--resume", type=Path, help="Resume from an existing last.pt checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = preflight(args.dataset_root, args.split_root)
    run_dir = args.project.resolve() / args.name
    config_path = run_dir / "dataset.generated.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_split(args.dataset_root, args.split_root / "train.txt", run_dir / "train.resolved.txt")
    write_resolved_split(args.dataset_root, args.split_root / "val.txt", run_dir / "val.resolved.txt")
    write_training_yaml(args.dataset_root, run_dir, config_path)

    report.update(
        {
            "model": args.model,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "fraction": args.fraction,
            "save_period": args.save_period,
            "device": args.device,
            "workers": args.workers,
            "seed": args.seed,
            "python": sys.version,
            "platform": platform.platform(),
            "dataset_yaml": str(config_path),
        }
    )
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.preflight_only:
        print("Preflight passed. No model was downloaded and no training was started.")
        return

    completed_artifacts = [run_dir / "results.csv", run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"]
    if args.resume is None and any(path.exists() for path in completed_artifacts):
        raise FileExistsError(
            f"Training artifacts already exist in {run_dir}. Choose a new --name so an earlier run is not overwritten."
        )

    from ultralytics import YOLO

    model = YOLO(str(args.resume.resolve()) if args.resume else args.model)
    train_args = {
        "data": str(config_path),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "fraction": args.fraction,
        "save_period": args.save_period,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "project": str(args.project.resolve()),
        "name": args.name,
        # The preflight report and generated YAML intentionally create this
        # directory before Ultralytics starts. Existing training artifacts are
        # rejected above.
        "exist_ok": True,
    }
    if args.device is not None:
        train_args["device"] = args.device
    if args.resume:
        train_args["resume"] = True
    model.train(**train_args)


if __name__ == "__main__":
    main()
