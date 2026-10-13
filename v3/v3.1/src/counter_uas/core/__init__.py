"""Stable records shared between independently replaceable modules."""

from counter_uas.core.detections import BoundingBox, Detection
from counter_uas.core.observations import Frame, SensorObservation

__all__ = ["BoundingBox", "Detection", "Frame", "SensorObservation"]

