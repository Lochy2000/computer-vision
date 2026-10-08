from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def image_signature(path: Path) -> tuple[np.ndarray, int, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    height, width = image.shape
    signature = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
    hash_image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = hash_image[:, 1:] > hash_image[:, :-1]
    hash_value = 0
    for bit in bits.flat:
        hash_value = (hash_value << 1) | int(bit)
    return signature, hash_value, (width, height)


def label_counts(path: Path) -> tuple[int, int]:
    counts = Counter(int(line.split()[0]) for line in path.read_text().splitlines() if line.strip())
    return counts[0], counts[1]


def build_features(root: Path, cache_path: Path) -> dict[str, np.ndarray]:
    names = sorted(path.name for path in (root / "images").glob("*.jpg"))
    signatures, hashes, dimensions, classes = [], [], [], []
    for name in names:
        signature, hash_value, dimension = image_signature(root / "images" / name)
        signatures.append(signature)
        hashes.append(hash_value)
        dimensions.append(dimension)
        classes.append(label_counts(root / "labels" / f"{Path(name).stem}.txt"))
    result = {
        "names": np.asarray(names),
        "signatures": np.asarray(signatures, dtype=np.uint8),
        "hashes": np.asarray(hashes, dtype=np.uint64),
        "dimensions": np.asarray(dimensions, dtype=np.int32),
        "classes": np.asarray(classes, dtype=np.int32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result


def load_or_build_features(root: Path, cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.exists():
        with np.load(cache_path) as cached:
            result = {key: cached[key] for key in cached.files}
        expected_names = sorted(path.name for path in (root / "images").glob("*.jpg"))
        if result["names"].tolist() == expected_names:
            return result
    return build_features(root, cache_path)


def cluster_scenes(features: dict[str, np.ndarray]) -> tuple[list[list[int]], dict]:
    raw = features["signatures"].reshape(len(features["names"]), -1).astype(np.float32) / 255.0
    means = raw.mean(axis=1, keepdims=True)
    centered = raw - means
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1e-8)
    hashes = features["hashes"]
    dimensions = features["dimensions"]
    union_find = UnionFind(len(hashes))
    candidate_pairs = accepted_pairs = 0
    accepted_correlations: list[float] = []
    for index in range(1, len(hashes)):
        distances = np.bitwise_count(np.bitwise_xor(hashes[index], hashes[:index]))
        candidates = np.flatnonzero(distances <= 22)
        candidate_pairs += len(candidates)
        if not len(candidates):
            continue
        correlations = normalized[candidates] @ normalized[index]
        maes = np.mean(np.abs(raw[candidates] - raw[index]), axis=1)
        same_dimensions = np.all(dimensions[candidates] == dimensions[index], axis=1)
        candidate_distances = distances[candidates]
        accepted = (
            (candidate_distances <= 18) & (correlations >= 0.88) & (maes <= 0.12)
        ) | ((correlations >= 0.85) & (maes <= 0.04))
        for candidate, correlation in zip(candidates[accepted], correlations[accepted]):
            union_find.union(index, int(candidate))
            accepted_pairs += 1
            accepted_correlations.append(float(correlation))
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(hashes)):
        members[union_find.find(index)].append(index)
    groups = sorted(members.values(), key=lambda group: (-len(group), features["names"][group[0]]))
    diagnostics = {
        "dhash_candidate_pairs": candidate_pairs,
        "accepted_similarity_edges": accepted_pairs,
        "groups": len(groups),
        "largest_group": len(groups[0]),
        "multi_image_groups": sum(len(group) > 1 for group in groups),
        "accepted_correlation_min": min(accepted_correlations, default=None),
    }
    return groups, diagnostics


def assign_groups(groups: list[list[int]], features: dict[str, np.ndarray]) -> dict[str, list[int]]:
    split_names = ("train", "val", "test")
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    total_images = len(features["names"])
    total_classes = features["classes"].sum(axis=0)
    total_groups = len(groups)
    targets = {
        split: np.asarray([total_images, *total_classes, total_groups], dtype=float) * ratio
        for split, ratio in ratios.items()
    }
    current = {split: np.zeros(4, dtype=float) for split in split_names}
    assignments: dict[str, list[int]] = {split: [] for split in split_names}
    for group in groups:
        addition = np.asarray([len(group), *features["classes"][group].sum(axis=0), 1], dtype=float)
        best_split = None
        best_score = None
        for split in split_names:
            simulated = {name: value.copy() for name, value in current.items()}
            simulated[split] += addition
            score = 0.0
            for name in split_names:
                scale = np.maximum(targets[name], 1.0)
                difference = (simulated[name] - targets[name]) / scale
                weights = np.asarray([1.0, 1.0, 1.0, 0.5])
                score += float(np.sum(weights * difference * difference))
                over = np.maximum(simulated[name] - targets[name] * 1.03, 0) / scale
                score += 6.0 * float(np.sum(over * over))
            if best_score is None or score < best_score:
                best_score, best_split = score, split
        assert best_split is not None
        assignments[best_split].extend(group)
        current[best_split] += addition
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene-grouped UB-SOD splits")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    features = load_or_build_features(args.root, args.cache)
    groups, diagnostics = cluster_scenes(features)
    assignments = assign_groups(groups, features)
    args.output.mkdir(parents=True, exist_ok=True)
    group_by_index = {}
    for group_id, group in enumerate(groups):
        for index in group:
            group_by_index[index] = group_id
    manifest = {
        "schema_version": 1,
        "method": {
            "name": "scene-v1",
            "dhash_candidate_distance": 22,
            "pixel_rule": "(dHash<=18 AND corr>=0.88 AND MAE<=0.12) OR (dHash<=22 AND corr>=0.85 AND MAE<=0.04)",
            "assignment": "whole connected components, greedy 80/10/10 image/UAV/Bird balance",
        },
        "diagnostics": diagnostics,
        "splits": {},
    }
    names = features["names"]
    for split, indices in assignments.items():
        ordered = sorted(indices, key=lambda index: names[index])
        relative_paths = [f"images/{names[index]}" for index in ordered]
        (args.output / f"{split}.txt").write_text("\n".join(relative_paths) + "\n", encoding="utf-8")
        class_totals = features["classes"][ordered].sum(axis=0)
        manifest["splits"][split] = {
            "images": len(ordered),
            "UAV_instances": int(class_totals[0]),
            "Bird_instances": int(class_totals[1]),
            "groups": len({group_by_index[index] for index in ordered}),
        }
    manifest["image_groups"] = {
        str(group_id): [str(names[index]) for index in group]
        for group_id, group in enumerate(groups)
    }
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostics": diagnostics, "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
