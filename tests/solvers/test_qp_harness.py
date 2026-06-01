import numpy as np
import pytest

from risk_aware_active_suspension.solvers.qp_harness import BoxQP


def test_unconstrained_quadratic_minimum() -> None:
    # min 0.5 u^T H u + q^T u, no active bounds -> u* = -H^{-1} q
    h = np.array([[4.0, 1.0], [1.0, 3.0]])
    qp = BoxQP(hessian=h, lower=np.full(2, -1e6), upper=np.full(2, 1e6))
    q = np.array([1.0, -2.0])
    qp.update_linear(q)
    res = qp.solve()
    expected = -np.linalg.solve(h, q)
    assert np.allclose(res.u, expected, atol=1e-6)
    assert res.status in ("solved", "solved inaccurate")


def test_box_bounds_clip_solution() -> None:
    h = np.eye(2)
    qp = BoxQP(hessian=h, lower=np.array([-0.5, -0.5]), upper=np.array([0.5, 0.5]))
    qp.update_linear(np.array([10.0, -10.0]))
    res = qp.solve()
    assert res.u[0] == pytest.approx(-0.5, abs=1e-6)
    assert res.u[1] == pytest.approx(0.5, abs=1e-6)


def test_invalid_dimensions_rejected() -> None:
    with pytest.raises(ValueError):
        BoxQP(hessian=np.zeros((2, 3)), lower=np.zeros(2), upper=np.zeros(2))
    with pytest.raises(ValueError):
        BoxQP(hessian=np.eye(2), lower=np.zeros(3), upper=np.zeros(2))


def test_warm_start_makes_repeated_solves_fast() -> None:
    h = 2.0 * np.eye(4)
    qp = BoxQP(hessian=h, lower=-np.ones(4), upper=np.ones(4))
    # Solve a sequence with slowly varying q; just check solver remains stable.
    for i in range(20):
        qp.update_linear(np.array([0.1 * i, -0.05 * i, 0.2, -0.1]))
        res = qp.solve()
        assert res.iterations <= 200
        assert res.solve_time_s < 0.01
