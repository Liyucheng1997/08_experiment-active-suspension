import numpy as np
import pytest

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_qp import (
    FullCarRiskAwareQP,
    RiskQPParams,
)
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.utils.config import LQRParams


@pytest.fixture
def default_lqr() -> LQRParams:
    return LQRParams()


@pytest.fixture
def comfort_equiv_risk_params() -> RiskWeightParams:
    # q_c ≡ 1, q_p ≡ 0 -> cost reduces to the unmodified comfort QP.
    return RiskWeightParams(
        rho_th=0.75, k_rho=25.0, kappa_rho=0.0,
        q_c_min=1.0, q_c_max=1.0,
        q_p_min=0.0, q_p_max=0.0,
    )


@pytest.fixture
def comfort_equiv_qp_params() -> RiskQPParams:
    # Disable rate constraint + zero R_u/R_du so risk-QP equals comfort-QP.
    return RiskQPParams(enforce_rate=False, r_u_factor=0.0, r_du_factor=0.0)


def test_acceptance_qp_zero_qp_collapses_to_comfort(
    default_vehicle, default_lqr, comfort_equiv_risk_params, comfort_equiv_qp_params
) -> None:
    """Plan acceptance: with q_p ≡ 0 and no margin constraint, matches Phase 3.4."""
    rqp = FullCarRiskAwareQP(default_vehicle, default_lqr,
                              comfort_equiv_risk_params, comfort_equiv_qp_params)
    cqp = FullCarComfortQP(default_vehicle, default_lqr)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.standard_normal(14) * np.array(
            [0.05, 0.5, 0.01, 0.1, 0.01, 0.1, *([0.005, 0.5] * 4)]
        )
        u_risk = rqp.compute(x)
        u_comf = cqp.compute(x)
        assert np.allclose(u_risk, u_comf, atol=2e-3), (u_risk, u_comf)


def test_acceptance_xi_is_zero_without_margin_constraint(default_vehicle, default_lqr) -> None:
    """Plan acceptance: with no margin constraint active, xi* = 0."""
    risk = RiskWeightParams(
        rho_th=0.75, k_rho=25.0, kappa_rho=5.0,
        q_c_min=0.1, q_c_max=1.0,
        q_p_min=0.0, q_p_max=100.0,
    )
    rqp = FullCarRiskAwareQP(default_vehicle, default_lqr, risk)
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.standard_normal(14) * 0.1
        rho = rng.uniform(0.5, 1.2, 4)
        rqp.compute(x, rho_ij=rho)
        assert np.all(np.abs(rqp.last_xi) < 1e-8), rqp.last_xi


def test_rate_constraint_binds_when_step_too_large(default_vehicle, default_lqr) -> None:
    # Tight rate: limit step to 50 N.
    risk = RiskWeightParams(q_c_max=1.0, q_c_min=1.0, q_p_min=0.0, q_p_max=0.0)
    qp = RiskQPParams(df_max=10000.0, enforce_rate=True)  # 10 kN/s * 5 ms = 50 N
    rqp = FullCarRiskAwareQP(default_vehicle, default_lqr, risk, qp)
    # Use a large state so the unconstrained optimum greatly exceeds 50 N.
    x = np.zeros(14)
    x[1] = 1.0
    u1 = rqp.compute(x)
    assert np.max(np.abs(u1)) <= 50.0 + 1e-6
    u2 = rqp.compute(x)
    # Second step can move another 50 N relative to the first.
    assert np.max(np.abs(u2 - u1)) <= 50.0 + 1e-6


def test_actuator_box_respected(default_vehicle) -> None:
    lqr = LQRParams(f_max=500.0)
    risk = RiskWeightParams(q_c_min=1.0, q_c_max=1.0, q_p_min=0.0, q_p_max=0.0)
    rqp = FullCarRiskAwareQP(default_vehicle, lqr, risk,
                              RiskQPParams(enforce_rate=False))
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = rng.standard_normal(14) * 2.0
        u = rqp.compute(x)
        assert np.all(u <= 500.0 + 1e-4)
        assert np.all(u >= -500.0 - 1e-4)


def test_state_shape_validation(default_vehicle, default_lqr, comfort_equiv_risk_params) -> None:
    rqp = FullCarRiskAwareQP(default_vehicle, default_lqr, comfort_equiv_risk_params)
    with pytest.raises(ValueError):
        rqp.compute(np.zeros(8))
    with pytest.raises(ValueError):
        rqp.compute(np.zeros(14), rho_ij=np.zeros(3))


def test_reset_clears_u_prev(default_vehicle, default_lqr, comfort_equiv_risk_params) -> None:
    rqp = FullCarRiskAwareQP(default_vehicle, default_lqr, comfort_equiv_risk_params)
    x = np.zeros(14); x[1] = 0.3
    rqp.compute(x)
    assert np.any(rqp._u_prev != 0.0)
    rqp.reset()
    assert np.all(rqp._u_prev == 0.0)


def test_solve_time_under_1ms(default_vehicle, default_lqr) -> None:
    risk = RiskWeightParams(q_c_min=0.1, q_c_max=1.0, q_p_min=0.0, q_p_max=10.0,
                            kappa_rho=5.0)
    rqp = FullCarRiskAwareQP(default_vehicle, default_lqr, risk)
    rng = np.random.default_rng(3)
    times = []
    for _ in range(200):
        x = rng.standard_normal(14) * 0.1
        rho = rng.uniform(0.0, 1.0, 4)
        rqp.compute(x, rho_ij=rho)
        times.append(rqp.last_solve_time_s)
    p95 = float(np.percentile(times, 95))
    assert p95 < 2.0e-3, f"p95 solve time = {p95*1000:.3f} ms"


def test_sigma_active_changes_q_c_and_q_p_local(default_vehicle, default_lqr) -> None:
    risk = RiskWeightParams(rho_th=0.75, k_rho=25.0, kappa_rho=5.0,
                            q_c_min=0.1, q_c_max=1.0,
                            q_p_min=0.0, q_p_max=10.0)
    rqp = FullCarRiskAwareQP(default_vehicle, default_lqr, risk)
    x = np.zeros(14); x[1] = 0.1
    rqp.compute(x, rho_ij=np.array([0.2, 0.2, 0.2, 0.2]))  # σ ≈ 0
    sigma_low = rqp.last_sigma
    qc_low = rqp.last_q_c
    rqp.compute(x, rho_ij=np.array([0.95, 0.95, 0.95, 0.95]))  # σ ≈ 1
    sigma_high = rqp.last_sigma
    qc_high = rqp.last_q_c
    assert sigma_low < 0.05
    assert sigma_high > 0.95
    assert qc_low > qc_high  # comfort weight fades as risk rises


def test_margin_constraint_activates_xi_when_authority_insufficient(default_vehicle, default_lqr) -> None:
    risk = RiskWeightParams(
        rho_th=0.75, k_rho=25.0, kappa_rho=5.0,
        q_c_min=0.8, q_c_max=1.0,
        q_p_min=1.0, q_p_max=5.0,
    )
    rqp = FullCarRiskAwareQP(
        default_vehicle,
        default_lqr,
        risk,
        RiskQPParams(gamma=-0.5, rho_safe=0.85, enforce_rate=False),
    )
    f_z = np.array([1000.0, 4000.0, 4000.0, 4000.0])
    f_c = np.array([1000.0, 0.0, 0.0, 0.0])

    u = rqp.compute(np.zeros(14), f_z_hat=f_z, f_c=f_c, mu=0.8)

    assert u.shape == (4,)
    assert rqp.last_xi[0] >= 0.0
    assert rqp.qp.gamma * u[0] + rqp.last_xi[0] >= f_c[0] / (0.8 * 0.85) - f_z[0] - 1e-3
