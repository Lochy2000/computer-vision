"""Full-person detection and short-lived anonymous tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from nanodet import NanoDet


PERSON_CLASS_ID = 0  # COCO class index


def _letterbox(frame: np.ndarray, size: int = 416) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize without stretching, returning top/left padding and resized dimensions."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = int(width * scale), int(height * scale)
    resized = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, (top, left, new_height, new_width)


class PersonDetector:
    """Return full-person boxes in the original frame's coordinates."""

    def __init__(self, model_path: Path, confidence: float = 0.4) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing person detector model: {model_path}")
        self.model = NanoDet(str(model_path), prob_threshold=confidence, iou_threshold=0.6)

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        input_frame, (top, left, new_height, new_width) = _letterbox(frame)
        predictions = self.model.infer(cv.cvtColor(input_frame, cv.COLOR_BGR2RGB))
        height, width = frame.shape[:2]
        boxes: list[np.ndarray] = []
        for prediction in predictions:
            if int(prediction[-1]) != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = prediction[:4]
            box = np.array([
                np.clip((x1 - left) * width / new_width, 0, width - 1),
                np.clip((y1 - top) * height / new_height, 0, height - 1),
                np.clip((x2 - left) * width / new_width, 0, width - 1),
                np.clip((y2 - top) * height / new_height, 0, height - 1),
            ], dtype=np.float32)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        return boxes


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


@dataclass
class TrackedPerson:
    track_id: int
    box: np.ndarray
    first_seen: datetime
    last_seen: datetime
    name: str = "Unknown"
    best_face_score: float = 0.0
    missed_detections: int = 0

    @property
    def label(self) -> str:
        return self.name if self.name != "Unknown" else f"Unknown #{self.track_id}"


class PersonTracker:
    """Greedy IoU tracker suitable for a stationary single-camera prototype."""

    def __init__(self, max_missed: int = 5, minimum_iou: float = 0.2) -> None:
        self.tracks: dict[int, TrackedPerson] = {}
        self.next_id = 1
        self.max_missed = max_missed
        self.minimum_iou = minimum_iou

    def update(self, boxes: list[np.ndarray], now: datetime) -> list[TrackedPerson]:
        available_tracks = set(self.tracks)
        available_boxes = set(range(len(boxes)))
        candidates = sorted(
            (
                (box_iou(track.box, box), track_id, box_index)
                for track_id, track in self.tracks.items()
                for box_index, box in enumerate(boxes)
            ),
            reverse=True,
        )

        for overlap, track_id, box_index in candidates:
            if overlap < self.minimum_iou:
                break
            if track_id not in available_tracks or box_index not in available_boxes:
                continue
            track = self.tracks[track_id]
            track.box = boxes[box_index]
            track.last_seen = now
            track.missed_detections = 0
            available_tracks.remove(track_id)
            available_boxes.remove(box_index)

        for track_id in available_tracks:
            self.tracks[track_id].missed_detections += 1
        for box_index in available_boxes:
            self.tracks[self.next_id] = TrackedPerson(self.next_id, boxes[box_index], now, now)
            self.next_id += 1

        return list(self.tracks.values())

    def pop_expired(self) -> list[TrackedPerson]:
        expired_ids = [
            track_id for track_id, track in self.tracks.items()
            if track.missed_detections > self.max_missed
        ]
        return [self.tracks.pop(track_id) for track_id in expired_ids]


def track_containing_face(face: np.ndarray, tracks: list[TrackedPerson]) -> TrackedPerson | None:
    """Choose the smallest person box containing the centre of a face."""
    x, y, width, height = face[:4]
    centre_x, centre_y = x + width / 2, y + height / 2
    containing = [
        track for track in tracks
        if track.box[0] <= centre_x <= track.box[2]
        and track.box[1] <= centre_y <= track.box[3]
    ]
    if not containing:
        return None
    return min(containing, key=lambda item: np.prod(item.box[2:] - item.box[:2]))
