import numpy as np
import pytest
from dataclasses import replace

from risk_aware_active_suspension.controllers.risk_qp_quarter import (
    QuarterCarRiskAwareQP,
    QuarterRiskQPParams,
)
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.metrics.tire import f_z_required as fz_required_fn
from risk_aware_active_suspension.metrics.tire import rho as rho_fn
from risk_aware_active_suspension.utils.config import LQRParams


@pytest.fixture
def quarter_risk_params() -> RiskWeightParams:
    return RiskWeightParams(
        rho_th=0.75, k_rho=25.0, kappa_rho=5.0,
        q_c_min=0.1, q_c_max=1.0,
        q_p_min=0.0, q_p_max=1.0,
    )


def test_margin_satisfied_when_authority_sufficient(default_vehicle, quarter_risk_params) -> None:
    """F_z + γ F_e ≥ F_z_required when actuator has enough authority."""
    lqr = LQRParams(f_max=10000.0)
    qp_params = QuarterRiskQPParams(gamma=1.0, rho_safe=0.85)
    qp = QuarterCarRiskAwareQP(default_vehicle, lqr, quarter_risk_params, qp_params)
    state = np.zeros(4)
    # F_z = 1500, demanded F_z_required ≈ 2614 -> gap of 1114 N; f_max=10kN.
    u = qp.compute(state, f_z_true=1500.0, f_c=2000.0, mu=0.9)
    assert qp.last_xi < 1.0, qp.last_xi
    assert 1500.0 + qp_params.gamma * float(u[0]) >= qp.last_f_z_required - 1e-2


def test_xi_quantifies_violation_when_authority_insufficient(
    default_vehicle, quarter_risk_params
) -> None:
    """When f_max ≪ required actuator force, ξ absorbs the remainder."""
    lqr = LQRParams(f_max=200.0)
    qp_params = QuarterRiskQPParams(gamma=1.0, rho_safe=0.85)
    qp = QuarterCarRiskAwareQP(default_vehicle, lqr, quarter_risk_params, qp_params)
    u = qp.compute(np.zeros(4), f_z_true=1500.0, f_c=2000.0, mu=0.9)
    assert abs(float(u[0])) == pytest.approx(200.0, abs=1.0)
    assert qp.last_xi > 100.0
    # ξ closes the gap exactly: F_z + γ·F_e + ξ = F_z_required.
    margin = 1500.0 + qp_params.gamma * float(u[0]) + qp.last_xi
    assert margin == pytest.approx(qp.last_f_z_required, abs=1e-2)


def test_no_margin_constraint_when_demand_below_capacity(
    default_vehicle, quarter_risk_params
) -> None:
    """When F_z >> F_z_required, controller behaves like comfort QP (ξ = 0, σ ≈ 0)."""
    qp = QuarterCarRiskAwareQP(default_vehicle, LQRParams(), quarter_risk_params)
    qp.compute(np.zeros(4), f_z_true=4000.0, f_c=500.0, mu=0.9)
    assert qp.last_xi < 1e-6
    assert qp.last_sigma < 0.05  # ρ ≈ 0.139 -> σ ≈ 0


def test_rho_safe_drives_required_load(default_vehicle, quarter_risk_params) -> None:
    """F_z_required scales as F_c / (μ ρ_safe)."""
    for rho_safe in (0.6, 0.85, 0.95):
        qp = QuarterCarRiskAwareQP(
            default_vehicle, LQRParams(), quarter_risk_params,
            QuarterRiskQPParams(rho_safe=rho_safe),
        )
        qp.compute(np.zeros(4), f_z_true=4000.0, f_c=1500.0, mu=0.9)
        expected = float(fz_required_fn(1500.0, 0.9, rho_safe))
        assert qp.last_f_z_required == pytest.approx(expected, rel=1e-6)


def test_gamma_zero_disables_actuator_influence_on_margin(
    default_vehicle, quarter_risk_params
) -> None:
    """With γ=0 the margin constraint can only be met by ξ."""
    qp = QuarterCarRiskAwareQP(
        default_vehicle, LQRParams(f_max=10000.0), quarter_risk_params,
        QuarterRiskQPParams(gamma=0.0, rho_safe=0.85),
    )
    u = qp.compute(np.zeros(4), f_z_true=1500.0, f_c=2000.0, mu=0.9)
    # F_e cannot contribute via the constraint -> ξ absorbs the full gap.
    expected_gap = fz_required_fn(2000.0, 0.9, 0.85) - 1500.0
    assert qp.last_xi == pytest.approx(expected_gap, abs=1.0)


def test_compute_rejects_invalid_inputs(default_vehicle, quarter_risk_params) -> None:
    qp = QuarterCarRiskAwareQP(default_vehicle, LQRParams(), quarter_risk_params)
    with pytest.raises(ValueError):
        qp.compute(np.zeros(3), f_z_true=4000.0, f_c=500.0, mu=0.9)
    with pytest.raises(ValueError):
        qp.compute(np.zeros(4), f_z_true=4000.0, f_c=500.0, mu=0.0)


def test_solve_time_under_1ms(default_vehicle, quarter_risk_params) -> None:
    qp = QuarterCarRiskAwareQP(default_vehicle, LQRParams(), quarter_risk_params)
    rng = np.random.default_rng(0)
    times = []
    for _ in range(200):
        f_z = float(np.clip(3700.0 + rng.standard_normal() * 800.0, 1000.0, 6000.0))
        qp.compute(rng.standard_normal(4) * 0.01, f_z_true=f_z, f_c=1500.0, mu=0.9)
        times.append(qp.last_solve_time_s)
    assert float(np.percentile(times, 95)) < 1.0e-3
