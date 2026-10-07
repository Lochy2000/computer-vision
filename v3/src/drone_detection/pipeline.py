from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from drone_detection.detectors.base import Detector
from drone_detection.types import FrameResult


def _capture_source(source: str) -> int | str:
    return int(source) if source.isdecimal() else source


def process_source(
    source: str | Path,
    detector: Detector,
    max_frames: int | None = None,
    frame_callback: Callable[[np.ndarray, FrameResult], None] | None = None,
) -> Iterator[FrameResult]:
    source_text = str(source)
    capture = cv2.VideoCapture(_capture_source(source_text))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source_text}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            started = perf_counter()
            detections = tuple(detector.predict(frame))
            inference_ms = (perf_counter() - started) * 1000.0
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_seconds = (
                timestamp_ms / 1000.0
                if timestamp_ms > 0
                else (frame_index / fps if fps > 0 else 0.0)
            )
            height, width = frame.shape[:2]
            result = FrameResult(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                source=source_text,
                frame_width=width,
                frame_height=height,
                inference_ms=inference_ms,
                detections=detections,
            )
            if frame_callback is not None:
                frame_callback(frame, result)
            yield result
            frame_index += 1
    finally:
        capture.release()

