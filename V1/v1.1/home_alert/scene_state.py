"""Pure scene-event state machine driven by timestamps and person counts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SceneState(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    POST_ROLL = "post_roll"


class SceneSignal(StrEnum):
    START_EVENT = "start_event"
    FINISH_EVENT = "finish_event"


@dataclass(frozen=True)
class SceneUpdate:
    previous_state: SceneState
    state: SceneState
    signals: tuple[SceneSignal, ...] = ()


class SceneStateMachine:
    """Decide when one continuous occupancy event starts and finishes.

    A person returning during post-roll continues the same event. When post-roll
    expires, ``FINISH_EVENT`` is emitted and the machine returns to ``IDLE``.
    Timestamps should come from ``time.monotonic()``.
    """

    def __init__(self, post_roll_seconds: float = 5.0) -> None:
        if post_roll_seconds < 0:
            raise ValueError("Post-roll duration cannot be negative")
        self.post_roll_seconds = post_roll_seconds
        self.state = SceneState.IDLE
        self.event_started_at: float | None = None
        self.last_person_seen_at: float | None = None
        self.post_roll_started_at: float | None = None
        self._latest_timestamp: float | None = None

    def update(self, timestamp: float, person_count: int) -> SceneUpdate:
        if person_count < 0:
            raise ValueError("Person count cannot be negative")
        if self._latest_timestamp is not None and timestamp < self._latest_timestamp:
            raise ValueError("Timestamps must be monotonic")
        self._latest_timestamp = timestamp

        previous = self.state
        signals: tuple[SceneSignal, ...] = ()

        if self.state is SceneState.IDLE:
            if person_count > 0:
                self.state = SceneState.RECORDING
                self.event_started_at = timestamp
                self.last_person_seen_at = timestamp
                signals = (SceneSignal.START_EVENT,)

        elif self.state is SceneState.RECORDING:
            if person_count > 0:
                self.last_person_seen_at = timestamp
            else:
                self.state = SceneState.POST_ROLL
                self.post_roll_started_at = timestamp
                if self._post_roll_has_elapsed(timestamp):
                    self._finish_event()
                    signals = (SceneSignal.FINISH_EVENT,)

        elif self.state is SceneState.POST_ROLL:
            if person_count > 0:
                self.state = SceneState.RECORDING
                self.last_person_seen_at = timestamp
                self.post_roll_started_at = None
            elif self._post_roll_has_elapsed(timestamp):
                self._finish_event()
                signals = (SceneSignal.FINISH_EVENT,)

        return SceneUpdate(previous, self.state, signals)

    def _post_roll_has_elapsed(self, timestamp: float) -> bool:
        if self.post_roll_started_at is None:
            raise RuntimeError("Post-roll state requires a start timestamp")
        return timestamp - self.post_roll_started_at >= self.post_roll_seconds

    def _finish_event(self) -> None:
        self.state = SceneState.IDLE
        self.event_started_at = None
        self.last_person_seen_at = None
        self.post_roll_started_at = None
