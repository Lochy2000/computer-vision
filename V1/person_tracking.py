"""NanoDet person detection adapted to the real ByteTrack tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

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

    def __init__(self, model_path: Path, confidence: float = 0.1) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing person detector model: {model_path}")
        self.model = NanoDet(str(model_path), prob_threshold=confidence, iou_threshold=0.6)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        input_frame, (top, left, new_height, new_width) = _letterbox(frame)
        predictions = self.model.infer(cv.cvtColor(input_frame, cv.COLOR_BGR2RGB))
        height, width = frame.shape[:2]
        detections: list[np.ndarray] = []
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
                detections.append(np.append(box, np.float32(prediction[-2])))
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(detections, dtype=np.float32)


@dataclass
class TrackedPerson:
    track_id: int
    box: np.ndarray
    first_seen: datetime
    last_seen: datetime
    name: str = "Unknown"
    best_face_score: float = 0.0
    identity_scores: dict[str, list[float]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name if self.name != "Unknown" else f"Unknown #{self.track_id}"

    def observe_identity(self, name: str, score: float, votes_required: int = 3) -> None:
        """Attach a name only after several consistent face matches."""
        if name == "Unknown":
            return
        scores = self.identity_scores.setdefault(name, [])
        scores.append(score)
        if len(scores) >= votes_required:
            winner, winner_scores = max(
                self.identity_scores.items(), key=lambda item: (len(item[1]), sum(item[1]))
            )
            self.name = winner
            self.best_face_score = max(winner_scores)


class PersonTracker:
    """Small adapter keeping application state around ByteTrack IDs."""

    def __init__(self, frame_rate: float = 30.0, expiry_seconds: float = 2.0) -> None:
        self.tracker = ByteTrackTracker(
            lost_track_buffer=max(1, round(expiry_seconds * 30)),
            frame_rate=max(frame_rate, 1.0),
            track_activation_threshold=0.4,
            high_conf_det_threshold=0.35,
            minimum_consecutive_frames=2,
            minimum_iou_threshold=0.1,
        )
        self.tracks: dict[int, TrackedPerson] = {}
        self.expiry_seconds = expiry_seconds

    def update(self, raw: np.ndarray, now: datetime) -> list[TrackedPerson]:
        detections = sv.Detections(
            xyxy=raw[:, :4],
            confidence=raw[:, 4],
            class_id=np.zeros(len(raw), dtype=int),
        )
        tracked = self.tracker.update(detections, timestamp=now.timestamp())
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        for box, tracker_id in zip(tracked.xyxy, tracker_ids):
            tracker_id = int(tracker_id)
            if tracker_id < 0:
                continue
            if tracker_id not in self.tracks:
                self.tracks[tracker_id] = TrackedPerson(tracker_id, box.copy(), now, now)
            else:
                person = self.tracks[tracker_id]
                person.box = box.copy()
                person.last_seen = now

        return list(self.tracks.values())

    def pop_expired(self, now: datetime) -> list[TrackedPerson]:
        expired_ids = [
            track_id for track_id, track in self.tracks.items()
            if (now - track.last_seen).total_seconds() > self.expiry_seconds
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
