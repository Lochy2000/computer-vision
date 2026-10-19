from __future__ import annotations


def require_identifier(value: str, field_name: str) -> str:
    """Validate an externally assigned, replay-stable identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value

