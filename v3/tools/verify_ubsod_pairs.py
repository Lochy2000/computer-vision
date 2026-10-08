from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def signature(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    return cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(left - right)))
    left_flat = left.ravel()
    right_flat = right.ravel()
    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    return mae, correlation


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify perceptual-hash UB-SOD pairs with direct pixels")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    pairs = audit["duplicates"]["holdout_near_train_dhash_le_4"]
    cache: dict[str, np.ndarray] = {}
    rows = []
    for pair in pairs:
        for name in (pair["holdout"], pair["train"]):
            if name not in cache:
                cache[name] = signature(args.images / name)
        mae, correlation = metrics(cache[pair["holdout"]], cache[pair["train"]])
        rows.append({**pair, "normalized_mae": mae, "pixel_correlation": correlation})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    thresholds = {
        "correlation_gte_0_95": sum(row["pixel_correlation"] >= 0.95 for row in rows),
        "correlation_gte_0_98": sum(row["pixel_correlation"] >= 0.98 for row in rows),
        "mae_lte_0_05": sum(row["normalized_mae"] <= 0.05 for row in rows),
        "mae_lte_0_02": sum(row["normalized_mae"] <= 0.02 for row in rows),
    }
    print(json.dumps({"pairs": len(rows), **thresholds}, indent=2))


if __name__ == "__main__":
    main()
