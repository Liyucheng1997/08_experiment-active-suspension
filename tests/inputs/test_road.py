import numpy as np
import pytest
from scipy.signal import welch

from risk_aware_active_suspension.inputs.road import (
    ISO8608_GD_N0,
    ISO8608_N0,
    iso8608,
    rounded_bump,
    sine,
    single_wheel_bump,
)


def test_sine_and_rounded_bump_shapes() -> None:
    t = np.linspace(0.0, 1.0, 101)

    assert sine(t, amplitude=0.1, freq_hz=1.0).shape == t.shape
    bump = rounded_bump(t, height=0.05, start=0.2, duration=0.4)
    assert bump.max() == pytest.approx(0.05, abs=1e-6)
    assert bump[0] == 0.0
    assert bump[-1] == 0.0


def test_single_wheel_bump_zero_pads_other_channels() -> None:
    t = np.linspace(0.0, 1.0, 101)
    channels = single_wheel_bump(t, wheel=2, height=0.04, start=0.2, duration=0.2)

    assert channels.shape == (len(t), 4)
    assert np.max(channels[:, 2]) > 0.0
    assert np.all(channels[:, [0, 1, 3]] == 0.0)


def test_iso8608_class_b_psd_has_expected_slope_and_level() -> None:
    v_x = 80.0 / 3.6
    dt = 0.002
    t = np.arange(0.0, 120.0, dt)
    road = iso8608("B", v_x=v_x, t=t, seed=2)

    freq, psd = welch(road, fs=1.0 / dt, nperseg=8192, scaling="density")
    mask = (freq >= 0.5) & (freq <= 20.0) & (psd > 0.0)
    slope, intercept = np.polyfit(np.log(freq[mask]), np.log(psd[mask]), 1)
    expected_psd = ISO8608_GD_N0["B"] * ((freq[mask] / v_x) / ISO8608_N0) ** -2 / v_x
    ratio = np.median(psd[mask] / expected_psd)

    assert slope == pytest.approx(-2.0, abs=0.35)
    assert ratio == pytest.approx(1.0, rel=0.6)
