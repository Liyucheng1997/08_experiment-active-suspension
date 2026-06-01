import numpy as np
import pytest

from risk_aware_active_suspension.controllers.lqr import (
    FullCarLQRController,
    QuarterCarLQRController,
)
from risk_aware_active_suspension.utils.config import LQRParams


@pytest.fixture
def default_lqr() -> LQRParams:
    return LQRParams()


def test_quarter_car_lqr_closed_loop_stable(default_vehicle, default_lqr) -> None:
    controller = QuarterCarLQRController(default_vehicle, default_lqr)
    assert controller.gain.shape == (1, 4)
    assert np.max(np.abs(controller.closed_loop_poles)) < 1.0


def test_quarter_car_lqr_zero_state_zero_force(default_vehicle, default_lqr) -> None:
    controller = QuarterCarLQRController(default_vehicle, default_lqr)
    assert controller.compute(np.zeros(4))[0] == 0.0


def test_quarter_car_lqr_saturation(default_vehicle) -> None:
    lqr = LQRParams(f_max=100.0)
    controller = QuarterCarLQRController(default_vehicle, lqr)
    state = np.array([0.0, 5.0, 0.0, 0.0])  # huge dz_s -> hits limit
    u = controller.compute(state)
    assert abs(u[0]) <= 100.0 + 1e-9


def test_full_car_lqr_closed_loop_stable(default_vehicle, default_lqr) -> None:
    controller = FullCarLQRController(default_vehicle, default_lqr)
    assert controller.gain.shape == (4, 14)
    assert np.max(np.abs(controller.closed_loop_poles)) < 1.0


def test_full_car_lqr_zero_state_zero_force(default_vehicle, default_lqr) -> None:
    controller = FullCarLQRController(default_vehicle, default_lqr)
    assert np.allclose(controller.compute(np.zeros(14)), 0.0)


def test_full_car_lqr_state_shape_validation(default_vehicle, default_lqr) -> None:
    controller = FullCarLQRController(default_vehicle, default_lqr)
    with pytest.raises(ValueError):
        controller.compute(np.zeros(8))


def test_lqr_params_loads_from_yaml(repo_root) -> None:
    from risk_aware_active_suspension.utils.config import lqr_from_yaml

    lqr = lqr_from_yaml(repo_root / "configs" / "controller_default.yaml")
    assert lqr.T_s == 0.005
    assert lqr.q_accel == 50.0
    assert lqr.q_phi == pytest.approx(5e6)
    assert lqr.r_force == pytest.approx(1e-5)
    assert lqr.f_max == 4000.0
