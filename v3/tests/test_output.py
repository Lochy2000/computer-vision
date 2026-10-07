import json

from drone_detection.output import JsonlWriter
from drone_detection.types import FrameResult


def test_jsonl_writer_writes_one_record(tmp_path) -> None:
    path = tmp_path / "result.jsonl"
    result = FrameResult(0, 0.0, "source", 640, 480, 1.2, ())
    with JsonlWriter(path) as writer:
        writer.write(result)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["frame_index"] == 0
    assert payload["detections"] == []

