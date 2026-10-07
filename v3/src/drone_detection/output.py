from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import numpy as np

from drone_detection.types import FrameResult


def annotate_frame(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Return a copy of a frame with all detections drawn on it."""
    import cv2

    rendered = frame.copy()
    for detection in result.detections:
        box = detection.bbox
        start = (round(box.x1), round(box.y1))
        end = (round(box.x2), round(box.y2))
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(rendered, start, end, (0, 255, 0), 2)
        cv2.putText(
            rendered,
            label,
            (start[0], max(20, start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return rendered


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


class AnnotatedVideoWriter:
    """Render detections to an MP4, initialising from the first frame."""

    def __init__(self, path: str | Path, fps: float = 30.0) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._writer = None

    def write(self, frame: np.ndarray, result: FrameResult) -> None:
        import cv2

        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not create annotated video: {self._path}")

        self._writer.write(annotate_frame(frame, result))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()

