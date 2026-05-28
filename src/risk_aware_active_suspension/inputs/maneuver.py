from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def straight(t: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(t, dtype=float)
    return time, np.zeros_like(time), np.zeros_like(time)


def step_steer(t: ArrayLike, steer_rad: float, start: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(t, dtype=float)
    delta_f = np.zeros_like(time)
    delta_f[time >= start] = steer_rad
    return time, np.zeros_like(time), delta_f


def j_turn(
    t: ArrayLike,
    steer_rad: float,
    start: float,
    rise_time: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rise_time <= 0.0:
        raise ValueError("rise_time must be positive.")
    time = np.asarray(t, dtype=float)
    tau = np.clip((time - start) / rise_time, 0.0, 1.0)
    smooth = tau * tau * (3.0 - 2.0 * tau)
    return time, np.zeros_like(time), steer_rad * smooth


def single_lane_change(
    t: ArrayLike,
    amplitude_rad: float,
    start: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(t, dtype=float)
    delta_f = _windowed_sine(time, amplitude_rad, start, duration, cycles=1.0)
    return time, np.zeros_like(time), delta_f


def double_lane_change(
    t: ArrayLike,
    amplitude_rad: float,
    start: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(t, dtype=float)
    delta_f = _windowed_sine(time, amplitude_rad, start, duration, cycles=2.0)
    return time, np.zeros_like(time), delta_f


def emergency_brake(
    t: ArrayLike,
    decel: float = 8.0,
    start: float = 0.0,
    rise_time: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if decel <= 0.0:
        raise ValueError("decel must be positive.")
    if rise_time <= 0.0:
        raise ValueError("rise_time must be positive.")
    time = np.asarray(t, dtype=float)
    tau = np.clip((time - start) / rise_time, 0.0, 1.0)
    smooth = tau * tau * (3.0 - 2.0 * tau)
    return time, -decel * smooth, np.zeros_like(time)


def _windowed_sine(
    t: np.ndarray,
    amplitude_rad: float,
    start: float,
    duration: float,
    cycles: float,
) -> np.ndarray:
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    signal = np.zeros_like(t)
    mask = (t >= start) & (t <= start + duration)
    tau = (t[mask] - start) / duration
    envelope = np.sin(np.pi * tau) ** 2
    signal[mask] = amplitude_rad * envelope * np.sin(2.0 * np.pi * cycles * tau)
    return signal
