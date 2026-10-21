from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationSetAudit:
    ready: bool
    image_count: int
    complete_annotation_count: int
    missing_files: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    duplicate_image_hashes: tuple[str, ...]
    invalid_records: tuple[str, ...]
    stratum_counts: dict[str, int]
    missing_strata: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "image_count": self.image_count,
            "complete_annotation_count": self.complete_annotation_count,
            "missing_files": list(self.missing_files),
            "hash_mismatches": list(self.hash_mismatches),
            "duplicate_image_hashes": list(self.duplicate_image_hashes),
            "invalid_records": list(self.invalid_records),
            "stratum_counts": self.stratum_counts,
            "missing_strata": list(self.missing_strata),
            "blockers": list(self.blockers),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_yolo_label(path: Path, class_ids: set[int]) -> str | None:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 5:
            return f"{path}: line {line_number} must contain 5 fields"
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            return f"{path}: line {line_number} contains a non-numeric value"
        if class_id not in class_ids:
            return f"{path}: line {line_number} has unknown class {class_id}"
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            return f"{path}: line {line_number} coordinates must be normalized"
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            return f"{path}: line {line_number} box must have positive size"
    return None


def audit_validation_manifest(
    manifest_path: str | Path,
    workspace_root: str | Path,
    policy: dict[str, Any],
) -> ValidationSetAudit:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(workspace_root)
    images = manifest.get("images")
    if not isinstance(images, list):
        raise ValueError("manifest images must be a list")
    aliases = policy.get("scenario_aliases", {})
    required_strata = tuple(policy["required_strata"])
    allowed_splits = set(policy.get("allowed_splits", ["validation"]))
    class_ids = {int(item["id"]) for item in manifest.get("classes", [])}

    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    invalid_records: list[str] = []
    hashes: list[str] = []
    strata: Counter[str] = Counter()
    completed = 0

    for index, record in enumerate(images):
        record_id = str(record.get("id", f"record-{index}"))
        image_value = record.get("image")
        image_hash = record.get("image_sha256")
        scenario = record.get("scenario")
        split = record.get("split")
        status = record.get("annotation_status")
        label_value = record.get("label")
        if not isinstance(image_value, str) or not isinstance(image_hash, str):
            invalid_records.append(f"{record_id}: image path and SHA-256 are required")
            continue
        image_path = root / image_value
        if not image_path.is_file():
            missing_files.append(image_value)
        elif _sha256(image_path) != image_hash:
            hash_mismatches.append(image_value)
        hashes.append(image_hash)
        canonical_scenario = aliases.get(scenario, scenario)
        if canonical_scenario in required_strata:
            strata[canonical_scenario] += 1
        else:
            invalid_records.append(f"{record_id}: unmapped scenario {scenario!r}")
        if split not in allowed_splits:
            invalid_records.append(f"{record_id}: split {split!r} is not allowed")
        if status != "complete":
            continue
        completed += 1
        if not isinstance(label_value, str):
            invalid_records.append(f"{record_id}: complete annotation requires label path")
            continue
        label_path = root / label_value
        if not label_path.is_file():
            missing_files.append(label_value)
            continue
        label_error = _validate_yolo_label(label_path, class_ids)
        if label_error:
            invalid_records.append(f"{record_id}: {label_error}")

    duplicate_hashes = tuple(sorted(value for value, count in Counter(hashes).items() if count > 1))
    missing_strata = tuple(name for name in required_strata if strata[name] == 0)
    blockers: list[str] = []
    if completed != len(images):
        blockers.append(f"{len(images) - completed} images lack complete human-reviewed annotations")
    if missing_strata:
        blockers.append("required scenario strata are missing")
    if missing_files:
        blockers.append("referenced files are missing")
    if hash_mismatches:
        blockers.append("image hashes do not match")
    if duplicate_hashes:
        blockers.append("duplicate image content is present")
    if invalid_records:
        blockers.append("manifest or label records are invalid")
    return ValidationSetAudit(
        ready=not blockers and bool(images),
        image_count=len(images),
        complete_annotation_count=completed,
        missing_files=tuple(sorted(missing_files)),
        hash_mismatches=tuple(sorted(hash_mismatches)),
        duplicate_image_hashes=duplicate_hashes,
        invalid_records=tuple(invalid_records),
        stratum_counts={name: strata[name] for name in required_strata},
        missing_strata=missing_strata,
        blockers=tuple(blockers),
    )
