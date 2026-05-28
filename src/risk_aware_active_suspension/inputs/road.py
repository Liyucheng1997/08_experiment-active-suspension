from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


ISO8608_GD_N0 = {
    "A": 16e-6,
    "B": 64e-6,
    "C": 256e-6,
    "D": 1024e-6,
    "E": 4096e-6,
}
ISO8608_N0 = 0.1


def sine(t: ArrayLike, amplitude: float, freq_hz: float, phase_rad: float = 0.0) -> np.ndarray:
    time = np.asarray(t, dtype=float)
    return amplitude * np.sin(2.0 * np.pi * freq_hz * time + phase_rad)


def rounded_bump(t: ArrayLike, height: float, start: float, duration: float) -> np.ndarray:
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    time = np.asarray(t, dtype=float)
    bump = np.zeros_like(time)
    mask = (time >= start) & (time <= start + duration)
    tau = (time[mask] - start) / duration
    bump[mask] = 0.5 * height * (1.0 - np.cos(2.0 * np.pi * tau))
    return bump


def iso8608(class_: str, v_x: float, t: ArrayLike, seed: int | None = None) -> np.ndarray:
    if v_x <= 0.0:
        raise ValueError("v_x must be positive.")
    key = class_.upper()
    if key not in ISO8608_GD_N0:
        raise ValueError(f"Unknown ISO 8608 class {class_!r}.")

    time = np.asarray(t, dtype=float)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("t must be a one-dimensional vector with at least two samples.")
    dt = float(np.median(np.diff(time)))
    if dt <= 0.0:
        raise ValueError("t must be strictly increasing.")

    n_samples = len(time)
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    spatial_freq = np.maximum(freqs / v_x, 1e-9)
    temporal_psd = ISO8608_GD_N0[key] * (spatial_freq / ISO8608_N0) ** -2 / v_x
    temporal_psd[0] = 0.0

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, len(freqs))
    spectrum = np.zeros(len(freqs), dtype=complex)
    if len(freqs) > 2:
        magnitude = np.sqrt(temporal_psd[1:-1] * fs * n_samples / 2.0)
        spectrum[1:-1] = magnitude * np.exp(1j * phase[1:-1])
    if n_samples % 2 == 0:
        spectrum[-1] = np.sqrt(temporal_psd[-1] * fs * n_samples) * np.cos(phase[-1])

    road = np.fft.irfft(spectrum, n=n_samples)
    return road - np.mean(road)


def single_wheel_bump(
    t: ArrayLike,
    wheel: int,
    height: float,
    start: float,
    duration: float,
) -> np.ndarray:
    if wheel not in (0, 1, 2, 3):
        raise ValueError("wheel must be one of 0, 1, 2, 3.")
    channels = np.zeros((len(np.asarray(t)), 4), dtype=float)
    channels[:, wheel] = rounded_bump(t, height=height, start=start, duration=duration)
    return channels
