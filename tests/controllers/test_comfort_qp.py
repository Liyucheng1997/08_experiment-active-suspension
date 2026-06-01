from dataclasses import replace

import numpy as np
import pytest

from risk_aware_active_suspension.controllers.comfort_qp import (
    FullCarComfortQP,
    QuarterCarComfortQP,
)
from risk_aware_active_suspension.controllers.lqr import (
    FullCarLQRController,
    QuarterCarLQRController,
)
from risk_aware_active_suspension.utils.config import LQRParams


@pytest.fixture
def unconstrained_lqr() -> LQRParams:
    return LQRParams(f_max=1.0e9)


@pytest.fixture
def bounded_lqr() -> LQRParams:
    return LQRParams(f_max=4000.0)


def test_quarter_unconstrained_qp_matches_lqr(default_vehicle, unconstrained_lqr) -> None:
    qp = QuarterCarComfortQP(default_vehicle, unconstrained_lqr)
    lqr = QuarterCarLQRController(default_vehicle, unconstrained_lqr)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal(4) * np.array([0.05, 0.5, 0.01, 0.5])
        assert np.allclose(qp.compute(x), lqr.compute(x), atol=1e-3)


def test_full_unconstrained_qp_matches_lqr(default_vehicle, unconstrained_lqr) -> None:
    qp = FullCarComfortQP(default_vehicle, unconstrained_lqr)
    lqr = FullCarLQRController(default_vehicle, unconstrained_lqr)
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.standard_normal(14) * np.array(
            [0.05, 0.5, 0.01, 0.1, 0.01, 0.1, *([0.005, 0.5] * 4)]
        )
        assert np.allclose(qp.compute(x), lqr.compute(x), atol=1e-3)


def test_full_qp_respects_actuator_bounds(default_vehicle) -> None:
    lqr = LQRParams(f_max=500.0)
    qp = FullCarComfortQP(default_vehicle, lqr)
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.standard_normal(14) * 2.0  # large state -> saturation
        u = qp.compute(x)
        assert np.all(u <= 500.0 + 1e-6)
        assert np.all(u >= -500.0 - 1e-6)


def test_full_qp_solve_time_p95_under_1ms(default_vehicle, bounded_lqr) -> None:
    qp = FullCarComfortQP(default_vehicle, bounded_lqr)
    rng = np.random.default_rng(3)
    times = []
    for _ in range(200):
        x = rng.standard_normal(14) * np.array(
            [0.05, 0.5, 0.01, 0.1, 0.01, 0.1, *([0.005, 0.5] * 4)]
        )
        qp.compute(x)
        times.append(qp.last_solve_time_s)
    p95 = float(np.percentile(times, 95))
    assert p95 < 1e-3, f"p95 solve time = {p95*1000:.3f} ms"


def test_state_shape_validation(default_vehicle, bounded_lqr) -> None:
    qq = QuarterCarComfortQP(default_vehicle, bounded_lqr)
    qf = FullCarComfortQP(default_vehicle, bounded_lqr)
    with pytest.raises(ValueError):
        qq.compute(np.zeros(3))
    with pytest.raises(ValueError):
        qf.compute(np.zeros(8))


def test_bounded_qp_still_matches_lqr_when_state_is_small(
    default_vehicle, bounded_lqr
) -> None:
    # If no constraint binds, bounded QP must still match LQR.
    qp = FullCarComfortQP(default_vehicle, bounded_lqr)
    lqr = FullCarLQRController(default_vehicle, replace(bounded_lqr, f_max=1e9))
    rng = np.random.default_rng(4)
    for _ in range(20):
        x = rng.standard_normal(14) * np.array(
            [0.001, 0.01, 0.0005, 0.005, 0.0005, 0.005, *([0.0005, 0.01] * 4)]
        )
        u_qp = qp.compute(x)
        u_lqr = lqr.compute(x)
        # Tiny state -> u far inside the box; should match.
        if np.max(np.abs(u_lqr)) < bounded_lqr.f_max * 0.5:
            assert np.allclose(u_qp, u_lqr, atol=1e-3)
