from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from drone_detection.types import FrameResult


class JsonlWriter:
    """Append one independently parseable benchmark record per frame."""

    def __init__(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = output_path.open("w", encoding="utf-8")

    def write(self, result: FrameResult) -> None:
        json.dump(result.to_dict(), self._file, separators=(",", ":"))
        self._file.write("\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

