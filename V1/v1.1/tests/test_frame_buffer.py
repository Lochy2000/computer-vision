from __future__ import annotations

import unittest

import numpy as np

from home_alert.frame_buffer import FrameBuffer


class FrameBufferTests(unittest.TestCase):
    def test_keeps_only_frames_inside_time_window(self) -> None:
        buffer = FrameBuffer(duration_seconds=2.0, maximum_bytes=1_000)
        for timestamp in (0.0, 1.0, 2.0, 3.0):
            buffer.append(np.full((2, 2), timestamp, dtype=np.float32), timestamp)

        frames = buffer.snapshot()
        self.assertEqual([frame.captured_at for frame in frames], [1.0, 2.0, 3.0])

    def test_memory_limit_evicts_oldest_frames(self) -> None:
        buffer = FrameBuffer(duration_seconds=10.0, maximum_bytes=8)
        buffer.append(np.zeros((2, 2), dtype=np.uint8), 1.0)
        buffer.append(np.ones((2, 2), dtype=np.uint8), 2.0)
        buffer.append(np.full((2, 2), 2, dtype=np.uint8), 3.0)

        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer.byte_size, 8)
        self.assertEqual([frame.captured_at for frame in buffer.snapshot()], [2.0, 3.0])

    def test_input_and_returned_frames_cannot_mutate_buffer(self) -> None:
        source = np.array([[7]], dtype=np.uint8)
        buffer = FrameBuffer(duration_seconds=5.0, maximum_bytes=100)
        buffer.append(source, 1.0)
        source[0, 0] = 8
        first_snapshot = buffer.snapshot()
        first_snapshot[0].image[0, 0] = 9

        self.assertEqual(int(buffer.snapshot()[0].image[0, 0]), 7)

    def test_rejects_timestamp_moving_backwards(self) -> None:
        buffer = FrameBuffer(duration_seconds=5.0, maximum_bytes=100)
        buffer.append(np.zeros((1, 1), dtype=np.uint8), 2.0)
        with self.assertRaises(ValueError):
            buffer.append(np.zeros((1, 1), dtype=np.uint8), 1.0)

    def test_zero_duration_disables_buffer(self) -> None:
        buffer = FrameBuffer(duration_seconds=0, maximum_bytes=100)
        buffer.append(np.zeros((1, 1), dtype=np.uint8), 1.0)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.byte_size, 0)


if __name__ == "__main__":
    unittest.main()
