"""Coordinate pre-roll buffering, scene state, and a recording destination."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .frame_buffer import BufferedFrame, FrameBuffer
from .scene_state import SceneSignal, SceneState, SceneStateMachine, SceneUpdate


class RecordingSink(Protocol):
    """Destination for one scene event's frames."""

    def start(self, frames: Sequence[BufferedFrame]) -> None: ...

    def append(self, frame: BufferedFrame) -> None: ...

    def finish(self) -> None: ...


@dataclass(frozen=True)
class CoordinatorUpdate:
    scene: SceneUpdate
    recording_active: bool


class EventCoordinator:
    """Turn frames and person counts into one continuous scene recording."""

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        scene_state: SceneStateMachine,
        recording_sink_factory: Callable[[], RecordingSink],
    ) -> None:
        self.frame_buffer = frame_buffer
        self.scene_state = scene_state
        self.recording_sink_factory = recording_sink_factory
        self._recording: RecordingSink | None = None

    def process_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        person_count: int,
    ) -> CoordinatorUpdate:
        """Process exactly one frame and any event transition it causes."""
        scene = self.scene_state.update(timestamp, person_count)
        self.frame_buffer.append(image, timestamp)
        current_frame = BufferedFrame(timestamp, np.ascontiguousarray(image).copy())

        if SceneSignal.START_EVENT in scene.signals:
            if self._recording is not None:
                raise RuntimeError("Cannot start an event while another recording is active")
            self._recording = self.recording_sink_factory()
            pre_roll = list(self.frame_buffer.snapshot())
            if not pre_roll or pre_roll[-1].captured_at != timestamp:
                pre_roll.append(current_frame)
            self._recording.start(pre_roll)
        elif scene.previous_state in (SceneState.RECORDING, SceneState.POST_ROLL):
            if self._recording is None:
                raise RuntimeError("Scene is active but no recording destination exists")
            self._recording.append(current_frame)

        if SceneSignal.FINISH_EVENT in scene.signals:
            if self._recording is None:
                raise RuntimeError("Cannot finish an event without an active recording")
            self._recording.finish()
            self._recording = None

        return CoordinatorUpdate(scene=scene, recording_active=self._recording is not None)
