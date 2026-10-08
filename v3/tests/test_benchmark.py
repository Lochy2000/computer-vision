from drone_detection.benchmark import summarise


def test_summarise_separates_cold_start() -> None:
    result = summarise([1000.0, 100.0, 100.0], [0, 1, 1])
    assert result["first_frame_ms"] == 1000.0
    assert result["steady_mean_inference_ms"] == 100.0
    assert result["steady_fps"] == 10.0
    assert result["frames_with_detections"] == 2

