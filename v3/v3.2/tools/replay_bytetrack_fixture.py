from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from counter_uas.evaluation import load_observation_sequence  # noqa: E402
from counter_uas.evaluation.recorded import sha256_file, summarize_replay  # noqa: E402
from counter_uas.tracking import ByteTrackConfig, ByteTrackTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one frozen fixture through ByteTrack.")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence = load_observation_sequence(args.fixture)
    config = ByteTrackConfig.from_json(args.config)
    tracker = ByteTrackTracker(config)
    frame_results = []
    all_tracks = []
    for batch in sequence.replay():
        tracks = tracker.update(batch.frame, batch.observations)
        all_tracks.append(tracks)
        frame_results.append(
            {
                "frame_id": batch.frame.frame_id,
                "sequence_number": batch.frame.sequence_number,
                "observation_ids": [item.observation_id for item in batch.observations],
                "tracks": [track.to_dict() for track in tracks],
            }
        )
    payload = {
        "checkpoint": "3.2.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": str(args.fixture),
        "fixture_sha256": sha256_file(args.fixture),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "tracker_id": tracker.tracker_id,
        "summary": summarize_replay(sequence, all_tracks),
        "frames": frame_results,
        "interpretation_limit": (
            "Descriptive replay only. Track presence is not ground-truth target matching."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
