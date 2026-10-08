from pathlib import Path
from types import SimpleNamespace

import pytest

from drone_detection.model_evaluation import canonical_hash, metrics_from_result, read_split


def test_canonical_hash_ignores_order_and_line_endings(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"images/b.jpg\r\nimages/a.jpg\r\n")
    second.write_bytes(b"images/a.jpg\nimages/b.jpg\n")
    assert canonical_hash(read_split(first)) == canonical_hash(read_split(second))


def test_read_split_rejects_duplicates(tmp_path: Path) -> None:
    split = tmp_path / "split.txt"
    split.write_text("images/a.jpg\nimages/a.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_split(split)


def test_metrics_from_result_is_json_ready() -> None:
    result = SimpleNamespace(
        names={0: "UAV", 1: "Bird"},
        box=SimpleNamespace(
            mp=0.8,
            mr=0.7,
            map50=0.75,
            map=0.5,
            p=[0.9, 0.7],
            r=[0.8, 0.6],
            ap50=[0.85, 0.65],
            ap=[0.6, 0.4],
        ),
        speed={"inference": 1.25},
        results_dict={"metrics/mAP50(B)": 0.75},
    )
    metrics = metrics_from_result(result)
    assert metrics["overall"]["map50_95"] == 0.5
    assert metrics["per_class"]["UAV"]["map50_95"] == 0.6
    assert metrics["per_class"]["Bird"]["map50_95"] == 0.4
