"""Typed configuration with safe, explicit defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RetentionConfig:
    known_days: int = 3
    unknown_days: int = 14
    uncertain_days: int = 14
    maximum_media_bytes: int = 5 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(days < 0 for days in (self.known_days, self.unknown_days, self.uncertain_days)):
            raise ValueError("Retention days cannot be negative")
        if self.maximum_media_bytes < 0:
            raise ValueError("Maximum media size cannot be negative")


@dataclass(frozen=True)
class AppConfig:
    data_directory: str = "data"
    camera_index: int = 0
    pre_roll_seconds: float = 5.0
    post_roll_seconds: float = 5.0
    maximum_pre_roll_bytes: int = 256 * 1024 * 1024
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("Camera index cannot be negative")
        if self.pre_roll_seconds < 0 or self.post_roll_seconds < 0:
            raise ValueError("Recording buffer durations cannot be negative")
        if self.maximum_pre_roll_bytes < 0:
            raise ValueError("Maximum pre-roll size cannot be negative")

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        retention = RetentionConfig(**raw.pop("retention", {}))
        return cls(retention=retention, **raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
