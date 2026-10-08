"""Local event-driven home monitoring foundation."""

from .config import AppConfig, RetentionConfig
from .coordinator import CoordinatorUpdate, EventCoordinator, RecordingSink
from .events import Event, EventState, IdentityKind
from .frame_buffer import BufferedFrame, FrameBuffer
from .repository import EventRepository
from .scene_state import SceneSignal, SceneState, SceneStateMachine, SceneUpdate

__all__ = [
    "AppConfig",
    "BufferedFrame",
    "CoordinatorUpdate",
    "Event",
    "EventRepository",
    "EventState",
    "EventCoordinator",
    "FrameBuffer",
    "IdentityKind",
    "RetentionConfig",
    "RecordingSink",
    "SceneSignal",
    "SceneState",
    "SceneStateMachine",
    "SceneUpdate",
]
