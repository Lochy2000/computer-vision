from __future__ import annotations

import unittest
from collections.abc import Sequence

import numpy as np

from home_alert.coordinator import EventCoordinator
from home_alert.frame_buffer import BufferedFrame, FrameBuffer
from home_alert.scene_state import SceneSignal, SceneStateMachine


class MemoryRecording:
    def __init__(self) -> None:
        self.frames: list[BufferedFrame] = []
        self.finished = False

    def start(self, frames: Sequence[BufferedFrame]) -> None:
        self.frames.extend(frames)

    def append(self, frame: BufferedFrame) -> None:
        self.frames.append(frame)

    def finish(self) -> None:
        self.finished = True


class RecordingFactory:
    def __init__(self) -> None:
        self.recordings: list[MemoryRecording] = []

    def __call__(self) -> MemoryRecording:
        recording = MemoryRecording()
        self.recordings.append(recording)
        return recording


def frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


class EventCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = RecordingFactory()
        self.coordinator = EventCoordinator(
            FrameBuffer(duration_seconds=2, maximum_bytes=1_000),
            SceneStateMachine(post_roll_seconds=2),
            self.factory,
        )

    def process(self, timestamp: float, people: int) -> None:
        self.coordinator.process_frame(frame(int(timestamp)), timestamp, people)

    def test_complete_event_contains_pre_roll_active_and_post_roll_frames(self) -> None:
        self.process(0, 0)
        self.process(1, 0)
        started = self.coordinator.process_frame(frame(2), 2, 1)
        self.process(3, 1)
        self.process(4, 0)
        finished = self.coordinator.process_frame(frame(6), 6, 0)

        recording = self.factory.recordings[0]
        self.assertIn(SceneSignal.START_EVENT, started.scene.signals)
        self.assertIn(SceneSignal.FINISH_EVENT, finished.scene.signals)
        self.assertEqual([item.captured_at for item in recording.frames], [0, 1, 2, 3, 4, 6])
        self.assertTrue(recording.finished)
        self.assertFalse(finished.recording_active)

    def test_return_during_post_roll_uses_same_recording(self) -> None:
        self.process(0, 1)
        self.process(1, 0)
        self.process(2, 1)
        self.process(3, 0)
        self.process(5, 0)

        self.assertEqual(len(self.factory.recordings), 1)
        self.assertTrue(self.factory.recordings[0].finished)

    def test_two_separated_visits_create_two_recordings(self) -> None:
        self.process(0, 1)
        self.process(1, 0)
        self.process(3, 0)
        self.process(10, 1)

        self.assertEqual(len(self.factory.recordings), 2)
        self.assertTrue(self.factory.recordings[0].finished)
        self.assertFalse(self.factory.recordings[1].finished)

    def test_recorded_frames_are_not_changed_by_caller_mutation(self) -> None:
        source = frame(7)
        self.coordinator.process_frame(source, 0, 1)
        source[:] = 99

        recorded = self.factory.recordings[0].frames[0].image
        self.assertTrue(np.all(recorded == 7))

    def test_zero_pre_roll_still_records_trigger_frame(self) -> None:
        factory = RecordingFactory()
        coordinator = EventCoordinator(
            FrameBuffer(duration_seconds=0, maximum_bytes=0),
            SceneStateMachine(post_roll_seconds=1),
            factory,
        )
        coordinator.process_frame(frame(5), 5, 1)

        self.assertEqual([item.captured_at for item in factory.recordings[0].frames], [5])

    def test_invalid_person_count_does_not_change_frame_buffer(self) -> None:
        with self.assertRaises(ValueError):
            self.coordinator.process_frame(frame(1), 1, -1)

        self.assertEqual(len(self.coordinator.frame_buffer), 0)
        self.assertEqual(len(self.factory.recordings), 0)


if __name__ == "__main__":
    unittest.main()
