from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from counter_uas.core import Frame, SensorObservation
from counter_uas.tracking.base import Tracker
from counter_uas.tracking.records import LocalTrack, TrackState


SUPPORTED_ULTRALYTICS_VERSION = "8.4.174"


@dataclass(frozen=True, slots=True)
class ByteTrackConfig:
    tracker_id: str = "ultralytics-bytetrack:3.2.2-v1"
    track_high_threshold: float = 0.25
    track_low_threshold: float = 0.10
    new_track_threshold: float = 0.25
    track_buffer_frames: int = 30
    match_threshold: float = 0.80
    fuse_score: bool = True
    minimum_confirmed_observations: int = 2
    tracked_class_ids: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not self.tracker_id.strip():
            raise ValueError("tracker_id must not be empty")
        thresholds = (
            self.track_low_threshold,
            self.track_high_threshold,
            self.new_track_threshold,
            self.match_threshold,
        )
        if not all(0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("ByteTrack thresholds must be between 0 and 1")
        if self.track_low_threshold > self.track_high_threshold:
            raise ValueError("track_low_threshold must not exceed track_high_threshold")
        if self.track_buffer_frames < 1:
            raise ValueError("track_buffer_frames must be at least 1")
        if self.minimum_confirmed_observations < 1:
            raise ValueError("minimum_confirmed_observations must be at least 1")
        if not self.tracked_class_ids:
            raise ValueError("tracked_class_ids must not be empty")
        if any(class_id < 0 for class_id in self.tracked_class_ids):
            raise ValueError("tracked_class_ids must be non-negative")
        if len(self.tracked_class_ids) != len(set(self.tracked_class_ids)):
            raise ValueError("tracked_class_ids must be unique")

    @classmethod
    def from_json(cls, path: str | Path) -> ByteTrackConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ByteTrack configuration root must be a JSON object")
        if "tracked_class_ids" in payload:
            payload["tracked_class_ids"] = tuple(payload["tracked_class_ids"])
        return cls(**payload)


@dataclass(slots=True)
class _TrackMemory:
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int
    missed_updates: int
    latest_observation: SensorObservation


class ByteTrackTracker(Tracker):
    """Ultralytics ByteTrack adapter for observation-only, single-sensor input."""

    def __init__(self, config: ByteTrackConfig | None = None) -> None:
        self._config = config or ByteTrackConfig()
        installed_version = version("ultralytics")
        if installed_version != SUPPORTED_ULTRALYTICS_VERSION:
            raise RuntimeError(
                "ByteTrack adapter requires ultralytics "
                f"{SUPPORTED_ULTRALYTICS_VERSION}; found {installed_version}"
            )

        try:
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack support is optional. Install with: "
                "python -m pip install -e '.[tracking]'"
            ) from exc

        backend_config = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=self._config.track_high_threshold,
            track_low_thresh=self._config.track_low_threshold,
            new_track_thresh=self._config.new_track_threshold,
            track_buffer=self._config.track_buffer_frames,
            match_thresh=self._config.match_threshold,
            fuse_score=self._config.fuse_score,
        )
        self._boxes_type = Boxes
        self._backend = BYTETracker(backend_config)
        self._sensor_id: str | None = None
        self._last_sequence_number: int | None = None
        self._memory: dict[int, _TrackMemory] = {}

    @property
    def tracker_id(self) -> str:
        return self._config.tracker_id

    def update(
        self,
        frame: Frame,
        observations: Sequence[SensorObservation],
    ) -> tuple[LocalTrack, ...]:
        self.validate_update(frame, observations)
        self._validate_stream_order(frame)
        candidates = tuple(
            observation
            for observation in observations
            if observation.detection.class_id in self._config.tracked_class_ids
        )
        boxes = self._boxes_type(
            self._boxes_array(candidates),
            (frame.height, frame.width),
        )
        backend_output = self._backend.update(boxes)

        active_track_ids: set[int] = set()
        snapshots: list[LocalTrack] = []
        for row in backend_output:
            backend_track_id = int(round(float(row[4])))
            observation_index = int(round(float(row[-1])))
            if not 0 <= observation_index < len(candidates):
                raise RuntimeError("ByteTrack returned an invalid observation index")
            observation = candidates[observation_index]
            active_track_ids.add(backend_track_id)
            memory = self._memory.get(backend_track_id)
            if memory is None:
                memory = _TrackMemory(
                    first_seen_at=observation.captured_at,
                    last_seen_at=observation.captured_at,
                    observation_count=1,
                    missed_updates=0,
                    latest_observation=observation,
                )
                self._memory[backend_track_id] = memory
            elif memory.latest_observation.observation_id != observation.observation_id:
                memory.last_seen_at = observation.captured_at
                memory.observation_count += 1
                memory.missed_updates = 0
                memory.latest_observation = observation
            snapshots.append(
                self._snapshot(backend_track_id, memory, self._active_state(memory))
            )

        for lost in self._backend.lost_stracks:
            backend_track_id = int(lost.track_id)
            if backend_track_id in active_track_ids:
                continue
            memory = self._memory.get(backend_track_id)
            if memory is None:
                continue
            memory.missed_updates += 1
            snapshots.append(
                self._snapshot(backend_track_id, memory, TrackState.LOST)
            )

        removed_ids = {int(track.track_id) for track in self._backend.removed_stracks}
        for backend_track_id in removed_ids:
            self._memory.pop(backend_track_id, None)

        self._sensor_id = frame.sensor_id
        self._last_sequence_number = frame.sequence_number
        return tuple(sorted(snapshots, key=lambda item: item.track_id))

    def reset(self) -> None:
        self._backend.reset()
        self._sensor_id = None
        self._last_sequence_number = None
        self._memory.clear()

    def _validate_stream_order(self, frame: Frame) -> None:
        if self._sensor_id is not None and frame.sensor_id != self._sensor_id:
            raise ValueError("ByteTrackTracker instance cannot mix sensor streams")
        if (
            self._last_sequence_number is not None
            and frame.sequence_number <= self._last_sequence_number
        ):
            raise ValueError("tracker frame sequence numbers must be strictly increasing")

    @staticmethod
    def _boxes_array(
        observations: tuple[SensorObservation, ...],
    ) -> np.ndarray:
        if not observations:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(
            [
                [
                    observation.detection.bbox.x1,
                    observation.detection.bbox.y1,
                    observation.detection.bbox.x2,
                    observation.detection.bbox.y2,
                    observation.detection.confidence,
                    observation.detection.class_id,
                ]
                for observation in observations
            ],
            dtype=np.float32,
        )

    def _active_state(self, memory: _TrackMemory) -> TrackState:
        if memory.observation_count >= self._config.minimum_confirmed_observations:
            return TrackState.CONFIRMED
        return TrackState.TENTATIVE

    def _snapshot(
        self,
        backend_track_id: int,
        memory: _TrackMemory,
        state: TrackState,
    ) -> LocalTrack:
        sensor_id = memory.latest_observation.sensor_id
        return LocalTrack(
            track_id=f"{sensor_id}:bytetrack:{backend_track_id}",
            tracker_id=self.tracker_id,
            sensor_id=sensor_id,
            state=state,
            first_seen_at=memory.first_seen_at,
            last_seen_at=memory.last_seen_at,
            observation_count=memory.observation_count,
            missed_updates=memory.missed_updates,
            latest_observation=memory.latest_observation,
        )

