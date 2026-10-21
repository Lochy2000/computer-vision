from __future__ import annotations

import hashlib
import json
from pathlib import Path

from counter_uas.evaluation.validation_set import audit_validation_manifest


def test_ready_manifest_requires_valid_hash_label_split_and_stratum(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"example-image")
    label = tmp_path / "image.txt"
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    manifest = {
        "classes": [{"id": 0, "name": "UAV"}],
        "images": [{
            "id": "example",
            "image": "image.jpg",
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "scenario": "close_drone",
            "split": "validation",
            "annotation_status": "complete",
            "label": "image.txt",
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_validation_manifest(
        path, tmp_path, {"required_strata": ["close_drone"], "allowed_splits": ["validation"]}
    )
    assert result.ready
    assert result.complete_annotation_count == 1


def test_pending_and_missing_strata_are_reported_without_fabricating_labels(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"example-image")
    manifest = {
        "classes": [{"id": 0, "name": "UAV"}],
        "images": [{
            "id": "pending",
            "image": "image.jpg",
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "scenario": "close_drone_meadow",
            "split": "unassigned",
            "annotation_status": "pending",
            "label": None,
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_validation_manifest(path, tmp_path, {
        "required_strata": ["close_drone", "bird"],
        "scenario_aliases": {"close_drone_meadow": "close_drone"},
        "allowed_splits": ["validation"],
    })
    assert not result.ready
    assert result.missing_strata == ("bird",)
    assert result.complete_annotation_count == 0
    assert any("split" in item for item in result.invalid_records)
