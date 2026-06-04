from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rms(values: ArrayLike) -> float:
    """Root-mean-square of a single signal (no reference)."""
    arr = _as_array(values)
    return float(np.sqrt(np.mean(arr**2)))


def peak_abs(values: ArrayLike) -> float:
    arr = _as_array(values)
    return float(np.max(np.abs(arr)))


def iae(t: ArrayLike, values: ArrayLike) -> float:
    time = _as_array(t)
    arr = _as_array(values)
    _validate_time_signal(time, arr)
    return float(np.trapezoid(np.abs(arr), time))


def itae(t: ArrayLike, values: ArrayLike) -> float:
    time = _as_array(t)
    arr = _as_array(values)
    _validate_time_signal(time, arr)
    return float(np.trapezoid(time * np.abs(arr), time))


def percentile95(values: ArrayLike) -> float:
    arr = _as_array(values)
    return float(np.percentile(np.abs(arr), 95))


def _as_array(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Metric input must not be empty.")
    return arr


def _validate_time_signal(t: np.ndarray, values: np.ndarray) -> None:
    if t.shape != values.shape:
        raise ValueError(f"Time and signal shapes must match, got {t.shape} and {values.shape}.")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("Time vector must be monotonic nondecreasing.")
