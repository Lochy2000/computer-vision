from __future__ import annotations

import json
from collections.abc import Sequence

from counter_uas.core import SensorObservation


def observations_to_json(
    observations: Sequence[SensorObservation],
    *,
    indent: int | None = None,
) -> str:
    """Return deterministic JSON text without performing file I/O."""
    payload = [observation.to_dict() for observation in observations]
    return json.dumps(
        payload,
        indent=indent,
        allow_nan=False,
        separators=None if indent is not None else (",", ":"),
    )

