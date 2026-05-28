from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


plt.rcParams.update(
    {
        "figure.figsize": (7.0, 4.0),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_timeseries(
    t: ArrayLike,
    signals: ArrayLike | Sequence[ArrayLike],
    labels: Sequence[str] | None = None,
    ylabel: str = "",
    title: str = "",
    save_path: str | Path | None = None,
):
    time = np.asarray(t, dtype=float)
    series = _normalize_series(signals)
    fig, ax = plt.subplots()
    for idx, values in enumerate(series):
        label = None if labels is None else labels[idx]
        ax.plot(time, values, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if labels is not None:
        ax.legend()
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig, ax


def plot_freq_response(
    freq_hz: ArrayLike,
    magnitude: ArrayLike,
    ylabel: str = "Magnitude",
    title: str = "",
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots()
    ax.semilogx(np.asarray(freq_hz, dtype=float), np.asarray(magnitude, dtype=float))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig, ax


def plot_side_by_side(
    t: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
    left_label: str,
    right_label: str,
    title: str = "",
    save_path: str | Path | None = None,
):
    time = np.asarray(t, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharex=True)
    axes[0].plot(time, np.asarray(left, dtype=float))
    axes[0].set_title(left_label)
    axes[0].set_xlabel("Time [s]")
    axes[1].plot(time, np.asarray(right, dtype=float))
    axes[1].set_title(right_label)
    axes[1].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig, axes


def _normalize_series(signals: ArrayLike | Sequence[ArrayLike]) -> list[np.ndarray]:
    arr = np.asarray(signals, dtype=float)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr[:, idx] for idx in range(arr.shape[1])]
    return [np.asarray(signal, dtype=float) for signal in signals]


def _save_if_requested(fig, save_path: str | Path | None) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
