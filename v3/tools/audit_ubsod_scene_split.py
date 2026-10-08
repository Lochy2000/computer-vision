from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def make_pair_montage(pairs: list[dict], image_root: Path, output: Path) -> None:
    rows = []
    for pair in pairs[:12]:
        cells = []
        for key in ("train", "holdout"):
            name = pair[key]
            image = cv2.imread(str(image_root / name))
            canvas = np.full((260, 480, 3), 245, np.uint8)
            scale = min(480 / image.shape[1], 225 / image.shape[0])
            resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            x = (480 - resized.shape[1]) // 2
            canvas[30 : 30 + resized.shape[0], x : x + resized.shape[1]] = resized
            label = f"{key}: {name} | corr={pair['correlation']:.3f} mae={pair['mae']:.3f}"
            cv2.putText(canvas, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (15, 15, 15), 1, cv2.LINE_AA)
            cells.append(canvas)
        rows.append(np.hstack(cells))
    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rebuilt UB-SOD scene splits")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.cache) as cached:
        features = {key: cached[key] for key in cached.files}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    name_to_index = {str(name): index for index, name in enumerate(features["names"])}
    split_by_name = {}
    group_by_name = {}
    for group_id, names in manifest["image_groups"].items():
        for name in names:
            group_by_name[name] = group_id
    split_lists = {}
    for split in ("train", "val", "test"):
        names = [Path(line).name for line in (args.manifest.parent / f"{split}.txt").read_text().splitlines() if line]
        split_lists[split] = names
        for name in names:
            if name in split_by_name:
                raise ValueError(f"Duplicate split assignment: {name}")
            split_by_name[name] = split
    missing = sorted(set(name_to_index) - set(split_by_name))
    unexpected = sorted(set(split_by_name) - set(name_to_index))
    group_splits = {}
    for name, split in split_by_name.items():
        group_splits.setdefault(group_by_name[name], set()).add(split)
    leaking_groups = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}

    raw = features["signatures"].reshape(len(features["names"]), -1).astype(np.float32) / 255.0
    centered = raw - raw.mean(axis=1, keepdims=True)
    normalized = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    hashes = features["hashes"]
    dimensions = features["dimensions"]
    train_indices = np.asarray([name_to_index[name] for name in split_lists["train"]], dtype=int)
    holdout_names = split_lists["val"] + split_lists["test"]
    primary_pairs = []
    sensitivity_pairs = []
    for name in holdout_names:
        index = name_to_index[name]
        distances = np.bitwise_count(np.bitwise_xor(hashes[index], hashes[train_indices]))
        candidate_mask = distances <= 22
        candidates = train_indices[candidate_mask]
        candidate_distances = distances[candidate_mask]
        if not len(candidates):
            continue
        correlations = normalized[candidates] @ normalized[index]
        maes = np.mean(np.abs(raw[candidates] - raw[index]), axis=1)
        primary = (
            (candidate_distances <= 18) & (correlations >= 0.88) & (maes <= 0.12)
        ) | ((correlations >= 0.85) & (maes <= 0.04))
        sensitivity = (correlations >= 0.85) & (maes <= 0.15)
        for candidate, correlation, mae in zip(candidates[primary], correlations[primary], maes[primary]):
            primary_pairs.append({"train": str(features["names"][candidate]), "holdout": name, "correlation": float(correlation), "mae": float(mae)})
        for candidate, correlation, mae in zip(candidates[sensitivity], correlations[sensitivity], maes[sensitivity]):
            sensitivity_pairs.append({"train": str(features["names"][candidate]), "holdout": name, "correlation": float(correlation), "mae": float(mae)})
    sensitivity_pairs.sort(key=lambda pair: (-pair["correlation"], pair["mae"]))
    split_stats = {}
    for split, names in split_lists.items():
        indices = [name_to_index[name] for name in names]
        classes = features["classes"][indices].sum(axis=0)
        split_stats[split] = {
            "images": len(names),
            "image_rate": len(names) / len(features["names"]),
            "UAV_instances": int(classes[0]),
            "Bird_instances": int(classes[1]),
            "groups": len({group_by_name[name] for name in names}),
        }
    report = {
        "schema_version": 1,
        "images": len(features["names"]),
        "assigned_images": len(split_by_name),
        "missing_images": missing,
        "unexpected_images": unexpected,
        "groups_crossing_splits": leaking_groups,
        "split_stats": split_stats,
        "primary_cross_split_similarity_pairs": primary_pairs,
        "sensitivity_cross_split_similarity_pairs": sensitivity_pairs,
        "decision": "pass" if not missing and not unexpected and not leaking_groups and not primary_pairs else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_pair_montage(sensitivity_pairs, args.image_root, args.output.parent / "sensitivity_pairs.jpg")
    print(json.dumps({
        "decision": report["decision"],
        "missing_images": len(missing),
        "unexpected_images": len(unexpected),
        "groups_crossing_splits": len(leaking_groups),
        "primary_pairs": len(primary_pairs),
        "sensitivity_pairs": len(sensitivity_pairs),
        "splits": split_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
