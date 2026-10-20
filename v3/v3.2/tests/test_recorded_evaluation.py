from __future__ import annotations

from pathlib import Path

import pytest

from counter_uas.evaluation import load_observation_sequence
from counter_uas.evaluation.recorded import summarize_observations, summarize_replay


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic-single-target-gap.json"


def test_observation_summary_preserves_empty_frames() -> None:
    sequence = load_observation_sequence(FIXTURE)
    assert summarize_observations(sequence) == {
        "frame_count": 3,
        "frames_with_observations": 2,
        "empty_frame_count": 1,
        "observation_count": 2,
        "class_counts": {"UAV": 2},
    }


def test_replay_summary_requires_one_track_result_per_frame() -> None:
    sequence = load_observation_sequence(FIXTURE)
    with pytest.raises(ValueError, match="one entry per fixture frame"):
        summarize_replay(sequence, [()])
