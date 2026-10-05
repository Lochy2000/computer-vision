"""A time- and memory-bounded buffer for pre-event video frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BufferedFrame:
    """One immutable point-in-time snapshot from the camera."""

    captured_at: float
    image: np.ndarray


class FrameBuffer:
    """Keep only the recent frames that may become an event's pre-roll.

    ``captured_at`` should come from ``time.monotonic()``. A byte limit protects
    the application from unexpectedly large camera resolutions or frame rates.
    """

    def __init__(self, duration_seconds: float, maximum_bytes: int) -> None:
        if duration_seconds < 0:
            raise ValueError("Buffer duration cannot be negative")
        if maximum_bytes < 0:
            raise ValueError("Buffer size cannot be negative")
        self.duration_seconds = duration_seconds
        self.maximum_bytes = maximum_bytes
        self._frames: deque[BufferedFrame] = deque()
        self._bytes = 0
        self._latest_timestamp: float | None = None

    def append(self, image: np.ndarray, captured_at: float) -> None:
        if self._latest_timestamp is not None and captured_at < self._latest_timestamp:
            raise ValueError("Frame timestamps must be monotonic")
        self._latest_timestamp = captured_at

        if self.duration_seconds == 0 or self.maximum_bytes == 0:
            self.clear()
            return

        snapshot = np.ascontiguousarray(image).copy()
        self._frames.append(BufferedFrame(captured_at, snapshot))
        self._bytes += snapshot.nbytes
        self._evict(captured_at)

    def snapshot(self) -> tuple[BufferedFrame, ...]:
        """Return independent copies safe for a recorder to consume."""
        return tuple(
            BufferedFrame(frame.captured_at, frame.image.copy()) for frame in self._frames
        )

    def clear(self) -> None:
        self._frames.clear()
        self._bytes = 0

    @property
    def byte_size(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._frames)

    def _evict(self, now: float) -> None:
        oldest_allowed = now - self.duration_seconds
        while self._frames and (
            self._frames[0].captured_at < oldest_allowed
            or self._bytes > self.maximum_bytes
        ):
            removed = self._frames.popleft()
            self._bytes -= removed.image.nbytes
