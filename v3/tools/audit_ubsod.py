from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "UAV", 1: "Bird"}
COLORS = {0: (0, 220, 0), 1: (0, 165, 255)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def load_splits(root: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    split_by_name: dict[str, str] = {}
    names_by_split: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        names = [Path(line.strip()).name for line in (root / f"{split}.txt").read_text().splitlines() if line.strip()]
        names_by_split[split] = names
        for name in names:
            if name in split_by_name:
                raise ValueError(f"Image appears in multiple splits: {name}")
            split_by_name[name] = split
    return split_by_name, names_by_split


def parse_labels(path: Path, width: int, height: int) -> tuple[list[dict], list[str]]:
    objects: list[dict] = []
    errors: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields")
            continue
        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric value")
            continue
        if class_id not in CLASS_NAMES:
            errors.append(f"line {line_number}: invalid class {class_id}")
            continue
        if not all(0 <= value <= 1 for value in (cx, cy, bw, bh)):
            errors.append(f"line {line_number}: normalized value outside [0, 1]")
        if bw <= 0 or bh <= 0:
            errors.append(f"line {line_number}: non-positive box size")
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        if x1 < -0.51 or y1 < -0.51 or x2 > width + 0.51 or y2 > height + 0.51:
            errors.append(f"line {line_number}: box extends outside image")
        objects.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width_px": bw * width,
                "height_px": bh * height,
                "area_px": bw * width * bh * height,
                "area_fraction": bw * bh,
            }
        )
    return objects, errors


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def make_montage(records: list[dict], output: Path, title: str, limit: int = 24) -> None:
    cells: list[np.ndarray] = []
    for record in records[:limit]:
        image = cv2.imread(record["image_path"])
        if image is None:
            continue
        for obj in record["objects"]:
            color = COLORS[obj["class_id"]]
            p1 = (max(0, round(obj["x1"])), max(0, round(obj["y1"])))
            p2 = (min(image.shape[1] - 1, round(obj["x2"])), min(image.shape[0] - 1, round(obj["y2"])))
            cv2.rectangle(image, p1, p2, color, max(2, round(min(image.shape[:2]) / 350)))
            cv2.putText(image, obj["class_name"], (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        canvas = np.full((270, 480, 3), 245, np.uint8)
        scale = min(480 / image.shape[1], 235 / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        x = (480 - resized.shape[1]) // 2
        canvas[30 : 30 + resized.shape[0], x : x + resized.shape[1]] = resized
        label = f"{record['name']} | {record['split']} | {len(record['objects'])} boxes"
        cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        cells.append(canvas)
    if not cells:
        return
    columns = 4
    rows = (len(cells) + columns - 1) // columns
    while len(cells) < rows * columns:
        cells.append(np.full_like(cells[0], 245))
    grid = np.vstack([np.hstack(cells[row * columns : (row + 1) * columns]) for row in range(rows)])
    header = np.full((55, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(header, title, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack([header, grid]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def run(root: Path, output_dir: Path) -> dict:
    split_by_name, names_by_split = load_splits(root)
    records: list[dict] = []
    validation_errors: list[dict] = []
    exact_hashes: dict[str, list[dict]] = defaultdict(list)
    dhashes: dict[int, list[dict]] = defaultdict(list)
    dimension_counts: Counter[str] = Counter()

    for name, split in sorted(split_by_name.items()):
        image_path = root / "images" / name
        label_path = root / "labels" / f"{Path(name).stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            validation_errors.append({"image": name, "error": "unreadable image"})
            continue
        height, width = image.shape[:2]
        dimension_counts[f"{width}x{height}"] += 1
        objects, errors = parse_labels(label_path, width, height)
        validation_errors.extend({"image": name, "error": error} for error in errors)
        record = {
            "name": name,
            "split": split,
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "objects": objects,
            "dhash": dhash(image),
        }
        records.append(record)
        exact_hashes[file_sha256(image_path)].append(record)
        dhashes[record["dhash"]].append(record)

    objects = [(record, obj) for record in records for obj in record["objects"]]
    class_counts = Counter(obj["class_name"] for _, obj in objects)
    split_class_counts: dict[str, Counter] = {split: Counter() for split in names_by_split}
    for record, obj in objects:
        split_class_counts[record["split"]][obj["class_name"]] += 1

    exact_cross_split = []
    for digest, group in exact_hashes.items():
        splits = sorted({item["split"] for item in group})
        if len(splits) > 1:
            exact_cross_split.append({"sha256": digest, "splits": splits, "images": [item["name"] for item in group]})

    same_dhash_cross_split = []
    for value, group in dhashes.items():
        splits = sorted({item["split"] for item in group})
        if len(splits) > 1:
            same_dhash_cross_split.append({"dhash": f"{value:016x}", "splits": splits, "images": [item["name"] for item in group]})

    train = [record for record in records if record["split"] == "train"]
    holdout = [record for record in records if record["split"] != "train"]
    near_train: list[dict] = []
    for record in holdout:
        nearest = min(train, key=lambda item: (record["dhash"] ^ item["dhash"]).bit_count())
        distance = (record["dhash"] ^ nearest["dhash"]).bit_count()
        if distance <= 4:
            near_train.append({"holdout": record["name"], "holdout_split": record["split"], "train": nearest["name"], "distance": distance})

    area_values = [obj["area_px"] for _, obj in objects]
    min_dimensions = [min(obj["width_px"], obj["height_px"]) for _, obj in objects]
    small_coco = sum(value < 32 * 32 for value in area_values)
    tiny_4 = sum(value < 4 for value in min_dimensions)
    tiny_8 = sum(value < 8 for value in min_dimensions)
    empty_images = sum(not record["objects"] for record in records)

    ranked = sorted(records, key=lambda record: min((obj["area_px"] for obj in record["objects"]), default=float("inf")))
    random_records = random.Random(20261008).sample(records, min(24, len(records)))
    near_names = {pair["holdout"] for pair in near_train[:12]} | {pair["train"] for pair in near_train[:12]}
    near_records = [record for record in records if record["name"] in near_names]
    make_montage(ranked, output_dir / "small_targets.jpg", "UB-SOD: smallest annotated targets")
    make_montage(random_records, output_dir / "stratified_random.jpg", "UB-SOD: deterministic random audit sample")
    make_montage(near_records, output_dir / "near_duplicate_candidates.jpg", "UB-SOD: cross-split perceptual-hash candidates")

    summary = {
        "dataset_root": str(root),
        "images": len(records),
        "split_images": {split: len(names) for split, names in names_by_split.items()},
        "instances": len(objects),
        "class_instances": dict(class_counts),
        "split_class_instances": {split: dict(counts) for split, counts in split_class_counts.items()},
        "empty_label_images": empty_images,
        "validation_errors": validation_errors,
        "dimensions_distinct": len(dimension_counts),
        "top_dimensions": dimension_counts.most_common(12),
        "object_size": {
            "area_px_p01": percentile(area_values, 1),
            "area_px_p10": percentile(area_values, 10),
            "area_px_median": percentile(area_values, 50),
            "area_px_p90": percentile(area_values, 90),
            "area_px_p99": percentile(area_values, 99),
            "coco_small_lt_1024": small_coco,
            "coco_small_rate": small_coco / len(objects),
            "min_dimension_lt_4px": tiny_4,
            "min_dimension_lt_8px": tiny_8,
        },
        "duplicates": {
            "exact_cross_split_groups": exact_cross_split,
            "same_dhash_cross_split_groups": same_dhash_cross_split,
            "holdout_near_train_dhash_le_4": near_train,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "objects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "split", "class_name", "width_px", "height_px", "area_px", "area_fraction"])
        writer.writeheader()
        for record, obj in objects:
            writer.writerow({"image": record["name"], "split": record["split"], **{key: obj[key] for key in writer.fieldnames if key not in {"image", "split"}}})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the UB-SOD release")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.root, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"validation_errors", "duplicates"}}, indent=2))
    print(f"validation_errors={len(summary['validation_errors'])}")
    print(f"exact_cross_split_groups={len(summary['duplicates']['exact_cross_split_groups'])}")
    print(f"same_dhash_cross_split_groups={len(summary['duplicates']['same_dhash_cross_split_groups'])}")
    print(f"near_train_pairs={len(summary['duplicates']['holdout_near_train_dhash_le_4'])}")


if __name__ == "__main__":
    main()
