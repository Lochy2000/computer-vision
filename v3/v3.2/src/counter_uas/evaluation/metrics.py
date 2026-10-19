from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from counter_uas.core.identifiers import require_identifier


@dataclass(frozen=True, slots=True)
class TargetFrameAssignment:
    """Reviewed assignment for one ground-truth target in one frame."""

    frame_id: str
    sequence_number: int
    target_visible: bool
    assigned_track_id: str | None
    unmatched_track_count: int = 0

    def __post_init__(self) -> None:
        require_identifier(self.frame_id, "frame_id")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        if self.assigned_track_id is not None:
            require_identifier(self.assigned_track_id, "assigned_track_id")
        if not self.target_visible and self.assigned_track_id is not None:
            raise ValueError("an absent target cannot have an assigned track")
        if self.unmatched_track_count < 0:
            raise ValueError("unmatched_track_count must be non-negative")


@dataclass(frozen=True, slots=True)
class IdentityMetrics:
    visible_frames: int
    tracked_visible_frames: int
    missed_visible_frames: int
    track_coverage: float
    identity_switches: int
    fragmentations: int
    unique_assigned_track_ids: int
    unmatched_track_instances: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_identity_metrics(
    assignments: tuple[TargetFrameAssignment, ...],
) -> IdentityMetrics:
    """Calculate transparent single-target POC tracking metrics.

    `track_coverage` is the share of visible target frames with an assigned
    track. A fragmentation is a track returning after one or more missed frames
    during the same continuous visibility episode. An identity switch is a
    change from the previous assigned ID within that visibility episode,
    including across a missed-frame gap.
    """
    frame_ids = [item.frame_id for item in assignments]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("assignment frame IDs must be unique")
    sequence_numbers = [item.sequence_number for item in assignments]
    if any(
        current <= previous
        for previous, current in zip(
            sequence_numbers,
            sequence_numbers[1:],
            strict=False,
        )
    ):
        raise ValueError("assignment sequence numbers must be strictly increasing")

    visible_frames = sum(item.target_visible for item in assignments)
    if visible_frames == 0:
        raise ValueError("at least one target-visible frame is required")
    tracked_visible_frames = sum(
        item.target_visible and item.assigned_track_id is not None
        for item in assignments
    )

    identity_switches = 0
    fragmentations = 0
    assigned_ids: set[str] = set()
    in_visibility_episode = False
    previously_tracked = False
    had_track_in_episode = False
    previous_assigned_id: str | None = None

    for item in assignments:
        if not item.target_visible:
            in_visibility_episode = False
            previously_tracked = False
            had_track_in_episode = False
            previous_assigned_id = None
            continue

        if not in_visibility_episode:
            in_visibility_episode = True
            previously_tracked = False
            had_track_in_episode = False
            previous_assigned_id = None

        current_id = item.assigned_track_id
        if current_id is None:
            previously_tracked = False
            continue

        assigned_ids.add(current_id)
        if had_track_in_episode and not previously_tracked:
            fragmentations += 1
        if previous_assigned_id is not None and current_id != previous_assigned_id:
            identity_switches += 1
        had_track_in_episode = True
        previously_tracked = True
        previous_assigned_id = current_id

    return IdentityMetrics(
        visible_frames=visible_frames,
        tracked_visible_frames=tracked_visible_frames,
        missed_visible_frames=visible_frames - tracked_visible_frames,
        track_coverage=tracked_visible_frames / visible_frames,
        identity_switches=identity_switches,
        fragmentations=fragmentations,
        unique_assigned_track_ids=len(assigned_ids),
        unmatched_track_instances=sum(
            item.unmatched_track_count for item in assignments
        ),
    )

