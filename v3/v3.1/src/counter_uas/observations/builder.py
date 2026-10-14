from __future__ import annotations

from collections.abc import Sequence

from counter_uas.core import Detection, Frame, SensorObservation


def _validate_detection_bounds(frame: Frame, detection: Detection) -> None:
    if detection.bbox.x2 > frame.width or detection.bbox.y2 > frame.height:
        raise ValueError(
            "detection bounding box exceeds Frame dimensions: "
            f"box=({detection.bbox.x1}, {detection.bbox.y1}, "
            f"{detection.bbox.x2}, {detection.bbox.y2}), "
            f"frame={frame.width}x{frame.height}"
        )


def build_observations(
    frame: Frame,
    detections: Sequence[Detection],
) -> tuple[SensorObservation, ...]:
    """Wrap ordered frame detections in deterministic observation records.

    Observation identity depends on the caller-provided sensor ID, frame ID,
    and detector output order. It is stable for an identical replay but should
    not be used as a persistent track identity.
    """
    observations: list[SensorObservation] = []
    for index, detection in enumerate(detections):
        _validate_detection_bounds(frame, detection)
        observations.append(
            SensorObservation(
                observation_id=(
                    f"{frame.sensor_id}:{frame.frame_id}:observation:{index:04d}"
                ),
                frame_id=frame.frame_id,
                sensor_id=frame.sensor_id,
                captured_at=frame.captured_at,
                detection=detection,
            )
        )
    return tuple(observations)

