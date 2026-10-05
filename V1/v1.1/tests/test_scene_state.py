from __future__ import annotations

import unittest

from home_alert.scene_state import SceneSignal, SceneState, SceneStateMachine


class SceneStateMachineTests(unittest.TestCase):
    def test_idle_scene_does_nothing_without_people(self) -> None:
        machine = SceneStateMachine(post_roll_seconds=5)
        result = machine.update(timestamp=0.0, person_count=0)
        self.assertEqual(result.state, SceneState.IDLE)
        self.assertEqual(result.signals, ())

    def test_first_person_starts_one_event(self) -> None:
        machine = SceneStateMachine(post_roll_seconds=5)
        first = machine.update(timestamp=1.0, person_count=1)
        second = machine.update(timestamp=2.0, person_count=2)

        self.assertEqual(first.state, SceneState.RECORDING)
        self.assertEqual(first.signals, (SceneSignal.START_EVENT,))
        self.assertEqual(second.state, SceneState.RECORDING)
        self.assertEqual(second.signals, ())
        self.assertEqual(machine.event_started_at, 1.0)

    def test_empty_scene_enters_post_roll_then_finishes(self) -> None:
        machine = SceneStateMachine(post_roll_seconds=5)
        machine.update(timestamp=1.0, person_count=1)
        waiting = machine.update(timestamp=2.0, person_count=0)
        still_waiting = machine.update(timestamp=6.9, person_count=0)
        finished = machine.update(timestamp=7.0, person_count=0)

        self.assertEqual(waiting.state, SceneState.POST_ROLL)
        self.assertEqual(still_waiting.state, SceneState.POST_ROLL)
        self.assertEqual(finished.state, SceneState.IDLE)
        self.assertEqual(finished.signals, (SceneSignal.FINISH_EVENT,))

    def test_return_during_post_roll_continues_same_event(self) -> None:
        machine = SceneStateMachine(post_roll_seconds=5)
        machine.update(timestamp=1.0, person_count=1)
        machine.update(timestamp=2.0, person_count=0)
        returned = machine.update(timestamp=4.0, person_count=1)

        self.assertEqual(returned.state, SceneState.RECORDING)
        self.assertEqual(returned.signals, ())
        self.assertEqual(machine.event_started_at, 1.0)

    def test_new_person_at_deadline_continues_event(self) -> None:
        machine = SceneStateMachine(post_roll_seconds=5)
        machine.update(timestamp=1.0, person_count=1)
        machine.update(timestamp=2.0, person_count=0)
        returned = machine.update(timestamp=7.0, person_count=1)

        self.assertEqual(returned.state, SceneState.RECORDING)
        self.assertNotIn(SceneSignal.FINISH_EVENT, returned.signals)

    def test_zero_post_roll_finishes_on_first_empty_update(self) -> None:
        machine = SceneStateMachine(post_roll_seconds=0)
        machine.update(timestamp=1.0, person_count=1)
        finished = machine.update(timestamp=2.0, person_count=0)

        self.assertEqual(finished.state, SceneState.IDLE)
        self.assertEqual(finished.signals, (SceneSignal.FINISH_EVENT,))

    def test_invalid_input_is_rejected(self) -> None:
        machine = SceneStateMachine()
        machine.update(timestamp=2.0, person_count=0)
        with self.assertRaises(ValueError):
            machine.update(timestamp=1.0, person_count=0)
        with self.assertRaises(ValueError):
            machine.update(timestamp=3.0, person_count=-1)


if __name__ == "__main__":
    unittest.main()
