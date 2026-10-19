from __future__ import annotations

import pytest

from counter_uas.evaluation import (
    TargetFrameAssignment,
    calculate_identity_metrics,
)


def assignment(
    frame: int,
    track_id: str | None,
    *,
    visible: bool = True,
    unmatched: int = 0,
) -> TargetFrameAssignment:
    return TargetFrameAssignment(str(frame), frame, visible, track_id, unmatched)


def test_metrics_expose_gap_fragmentation_and_identity_change() -> None:
    metrics = calculate_identity_metrics(
        (
            assignment(0, "track-1"),
            assignment(1, "track-1"),
            assignment(2, None),
            assignment(3, "track-2", unmatched=2),
            assignment(4, "track-2"),
        )
    )
    assert metrics.visible_frames == 5
    assert metrics.tracked_visible_frames == 4
    assert metrics.missed_visible_frames == 1
    assert metrics.track_coverage == 0.8
    assert metrics.fragmentations == 1
    assert metrics.identity_switches == 1
    assert metrics.unique_assigned_track_ids == 2
    assert metrics.unmatched_track_instances == 2


def test_target_absence_starts_a_new_visibility_episode() -> None:
    metrics = calculate_identity_metrics(
        (
            assignment(0, "track-1"),
            assignment(1, None, visible=False, unmatched=1),
            assignment(2, "track-2"),
        )
    )
    assert metrics.fragmentations == 0
    assert metrics.identity_switches == 0
    assert metrics.unique_assigned_track_ids == 2
    assert metrics.unmatched_track_instances == 1


def test_metric_input_rejects_duplicate_frame_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        calculate_identity_metrics((assignment(0, "a"), assignment(0, "a")))


def test_metric_input_rejects_non_increasing_sequence() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        calculate_identity_metrics((assignment(1, "a"), assignment(0, "a")))


def test_metric_requires_visible_ground_truth() -> None:
    with pytest.raises(ValueError, match="target-visible"):
        calculate_identity_metrics((assignment(0, None, visible=False),))


def test_absent_target_cannot_receive_assignment() -> None:
    with pytest.raises(ValueError, match="absent"):
        assignment(0, "track-1", visible=False)

