from pathlib import Path

from drone_detection.evalset import evenly_spaced_indices, validate_yolo_label


def test_evenly_spaced_indices_include_first_and_last() -> None:
    assert evenly_spaced_indices(101, 3) == [0, 50, 100]
    assert evenly_spaced_indices(2, 10) == [0, 1]


def test_validate_yolo_label(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    assert validate_yolo_label(label) == []

    label.write_text("1 0.5 0.5 0.0 1.2\n", encoding="utf-8")
    errors = validate_yolo_label(label)
    assert len(errors) == 3
