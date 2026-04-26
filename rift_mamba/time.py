"""Time helpers shared by causal extraction and feature encoding."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


def parse_time(value: Any) -> datetime | None:
    """Parse common timestamp representations into a timezone-naive datetime."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    raise TypeError(f"cannot parse timestamp from {type(value).__name__}")


def days_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 86_400.0


def fourier_time_features(
    delta_days: float,
    periods: tuple[float, ...] = (1.0, 7.0, 30.0, 365.0),
) -> np.ndarray:
    """Encode a relative time delta with sin/cos periodic features."""

    features: list[float] = []
    for period in periods:
        if period <= 0:
            raise ValueError("Fourier time periods must be positive")
        angle = 2.0 * np.pi * delta_days / period
        features.extend([float(np.sin(angle)), float(np.cos(angle))])
    return np.asarray(features, dtype=np.float32)
