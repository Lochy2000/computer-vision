from dataclasses import dataclass

import pytest

from counter_uas.core import BoundingBox, Detection
from counter_uas.verification import compare_detections


def detection(*, confidence: float = 0.8, class_name: str = "UAV") -> Detection:
    return Detection(BoundingBox(1, 2, 10, 12), confidence, 0, class_name, "model")


def test_equivalent_detections_have_no_differences() -> None:
    assert compare_detections((detection(),), (detection(),)) == ()


def test_tolerance_absorbs_insignificant_float_difference() -> None:
    differences = compare_detections(
        (detection(confidence=0.8),),
        (detection(confidence=0.800001),),
        absolute_tolerance=1e-5,
    )
    assert differences == ()


def test_material_difference_is_reported() -> None:
    differences = compare_detections(
        (detection(class_name="UAV"),),
        (detection(class_name="Bird"), detection()),
    )
    assert {difference.field for difference in differences} == {"count", "class_name"}


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compare_detections((), (), absolute_tolerance=-1)

