import numpy as np
import pytest

from risk_aware_active_suspension.controllers.passive import PassiveController
from risk_aware_active_suspension.inputs.road import rounded_bump, iso8608
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.plants.quarter_car import QuarterCar


def test_compute_returns_zero_vector_of_requested_size() -> None:
    controller = PassiveController(n_actuators=4)
    u = controller.compute(np.zeros(14))

    assert u.shape == (4,)
    assert u.dtype == np.float64
    assert np.all(u == 0.0)


def test_compute_is_independent_of_state_and_extra_args() -> None:
    controller = PassiveController(n_actuators=4)
    rng = np.random.default_rng(0)

    for _ in range(5):
        state = rng.standard_normal(14)
        u = controller.compute(state, t=1.23, road=np.ones(4), fz_hat=np.full(4, 4000.0))
        assert np.all(u == 0.0)


def test_compute_returns_a_copy_so_callers_cannot_mutate_internal_state() -> None:
    controller = PassiveController(n_actuators=4)
    u = controller.compute(np.zeros(14))
    u[:] = 999.0
    assert np.all(controller.compute(np.zeros(14)) == 0.0)


def test_non_positive_n_actuators_rejected() -> None:
    with pytest.raises(ValueError):
        PassiveController(n_actuators=0)


def test_closed_loop_full_car_is_bit_identical_to_open_loop(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    controller = PassiveController(n_actuators=4)
    t = np.arange(0.0, 4.0, 0.001)
    bump = rounded_bump(t, height=0.02, start=0.5, duration=1.0)
    roads = np.column_stack([bump, np.zeros_like(bump), bump, np.zeros_like(bump)])

    open_loop = plant.simulate(t, roads)

    closed_loop = np.zeros_like(open_loop)
    state = np.zeros(14)
    for k in range(len(t) - 1):
        u = controller.compute(state, t=float(t[k]), road=roads[k])
        assert np.all(u == 0.0)
        state = plant.step(state, roads[k], float(t[k + 1] - t[k]))
        closed_loop[k + 1] = state

    assert np.array_equal(closed_loop, open_loop)


def test_closed_loop_quarter_car_on_iso_road_is_bit_identical(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    controller = PassiveController(n_actuators=1)
    t = np.arange(0.0, 5.0, 0.001)
    road = iso8608("B", v_x=80.0 / 3.6, t=t, seed=3)

    open_loop = plant.simulate(t, u_seq=0.0, w_seq=road)

    closed_loop = np.zeros_like(open_loop)
    state = np.zeros(4)
    for k in range(len(t) - 1):
        u = controller.compute(state, w=float(road[k]))
        state = plant.step(state, float(u[0]), float(road[k]), float(t[k + 1] - t[k]))
        closed_loop[k + 1] = state

    assert np.array_equal(closed_loop, open_loop)
