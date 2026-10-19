from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from counter_uas.core import Frame, SensorObservation
from counter_uas.tracking.records import LocalTrack


class Tracker(ABC):
    """Contract for a replaceable, single-sensor tracking implementation."""

    @property
    @abstractmethod
    def tracker_id(self) -> str:
        """Stable identifier for the tracker implementation and configuration."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        frame: Frame,
        observations: Sequence[SensorObservation],
    ) -> tuple[LocalTrack, ...]:
        """Advance tracker state by exactly one frame.

        Empty observations are valid and allow implementations to age tracks.
        Returned records are the reportable track snapshots for this update.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracker state so a replay can start deterministically."""
        raise NotImplementedError

    @staticmethod
    def validate_update(
        frame: Frame,
        observations: Sequence[SensorObservation],
    ) -> None:
        observation_ids: set[str] = set()
        for observation in observations:
            if observation.observation_id in observation_ids:
                raise ValueError("observation IDs must be unique within an update")
            observation_ids.add(observation.observation_id)
            if observation.sensor_id != frame.sensor_id:
                raise ValueError("observation sensor_id does not match Frame")
            if observation.frame_id != frame.frame_id:
                raise ValueError("observation frame_id does not match Frame")
            if observation.captured_at != frame.captured_at:
                raise ValueError("observation captured_at does not match Frame")

