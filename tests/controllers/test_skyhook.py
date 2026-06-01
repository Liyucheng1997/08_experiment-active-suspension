import numpy as np
import pytest

from risk_aware_active_suspension.controllers.skyhook import (
    FullCarSkyhookController,
    QuarterCarSkyhookController,
)


def test_quarter_car_skyhook_force_formula() -> None:
    controller = QuarterCarSkyhookController(c_sky=2000.0, f_max=None)
    state = np.array([0.0, 0.3, 0.0, 0.0])  # dot z_s = 0.3 m/s
    u = controller.compute(state)
    assert u.shape == (1,)
    assert u[0] == pytest.approx(-600.0)


def test_quarter_car_saturation_clips_both_signs() -> None:
    controller = QuarterCarSkyhookController(c_sky=10000.0, f_max=1000.0)
    assert controller.compute(np.array([0.0, 1.0, 0.0, 0.0]))[0] == pytest.approx(-1000.0)
    assert controller.compute(np.array([0.0, -1.0, 0.0, 0.0]))[0] == pytest.approx(1000.0)


def test_quarter_car_zero_velocity_yields_zero_force() -> None:
    controller = QuarterCarSkyhookController(c_sky=5000.0)
    assert controller.compute(np.zeros(4))[0] == 0.0


def test_full_car_skyhook_uses_per_corner_sprung_velocity(default_vehicle) -> None:
    controller = FullCarSkyhookController(
        params=default_vehicle, c_sky=1000.0, f_max=None
    )
    # Pure roll velocity dot phi = 0.1 -> corner velocities = y_ij * 0.1
    state = np.zeros(14)
    state[3] = 0.1
    u = controller.compute(state)
    half_t = default_vehicle.t / 2.0
    expected = -1000.0 * np.array(
        [half_t, -half_t, half_t, -half_t]
    ) * 0.1
    assert np.allclose(u, expected)


def test_full_car_skyhook_pure_pitch_velocity(default_vehicle) -> None:
    controller = FullCarSkyhookController(
        params=default_vehicle, c_sky=2000.0, f_max=None
    )
    state = np.zeros(14)
    state[5] = 0.2  # dot theta
    u = controller.compute(state)
    expected = -2000.0 * np.array(
        [default_vehicle.l_f, default_vehicle.l_f, -default_vehicle.l_r, -default_vehicle.l_r]
    ) * 0.2
    assert np.allclose(u, expected)


def test_full_car_skyhook_pure_heave_velocity(default_vehicle) -> None:
    controller = FullCarSkyhookController(
        params=default_vehicle, c_sky=1500.0, f_max=None
    )
    state = np.zeros(14)
    state[1] = 0.4
    u = controller.compute(state)
    assert np.allclose(u, np.full(4, -1500.0 * 0.4))


def test_full_car_saturation(default_vehicle) -> None:
    controller = FullCarSkyhookController(
        params=default_vehicle, c_sky=1e6, f_max=2500.0
    )
    state = np.zeros(14)
    state[1] = 1.0
    u = controller.compute(state)
    assert np.all(u == -2500.0)


def test_state_shape_validation(default_vehicle) -> None:
    q = QuarterCarSkyhookController(c_sky=1000.0)
    with pytest.raises(ValueError):
        q.compute(np.zeros(3))
    f = FullCarSkyhookController(params=default_vehicle, c_sky=1000.0)
    with pytest.raises(ValueError):
        f.compute(np.zeros(8))
