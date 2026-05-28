import numpy as np
import pytest

from risk_aware_active_suspension.inputs.maneuver import (
    double_lane_change,
    emergency_brake,
    j_turn,
    single_lane_change,
    step_steer,
    straight,
)


def test_straight_returns_zero_inputs() -> None:
    t = np.linspace(0.0, 1.0, 11)
    out_t, a_x, delta_f = straight(t)

    assert np.array_equal(out_t, t)
    assert np.all(a_x == 0.0)
    assert np.all(delta_f == 0.0)


def test_step_steer_and_j_turn() -> None:
    t = np.linspace(0.0, 2.0, 201)
    _, _, step = step_steer(t, steer_rad=0.1, start=1.0)
    _, _, smooth = j_turn(t, steer_rad=0.1, start=1.0, rise_time=0.5)

    assert step[t < 1.0].max() == 0.0
    assert step[-1] == pytest.approx(0.1)
    assert smooth[-1] == pytest.approx(0.1)
    assert smooth[np.searchsorted(t, 1.1)] < 0.1


def test_lane_change_profiles_are_windowed() -> None:
    t = np.linspace(0.0, 5.0, 501)
    _, _, single = single_lane_change(t, amplitude_rad=0.08, start=1.0, duration=2.0)
    _, _, double = double_lane_change(t, amplitude_rad=0.08, start=1.0, duration=2.0)

    assert np.all(single[t < 1.0] == 0.0)
    assert np.all(single[t > 3.0] == 0.0)
    assert np.count_nonzero(np.diff(np.signbit(single))) < np.count_nonzero(np.diff(np.signbit(double)))


def test_emergency_brake_is_negative_acceleration() -> None:
    t = np.linspace(0.0, 1.0, 101)
    _, a_x, delta_f = emergency_brake(t, decel=8.0, start=0.2, rise_time=0.2)

    assert a_x[0] == 0.0
    assert a_x[-1] == pytest.approx(-8.0)
    assert np.all(delta_f == 0.0)
