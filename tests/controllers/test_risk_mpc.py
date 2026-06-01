import numpy as np
import pytest

from risk_aware_active_suspension.controllers.risk_mpc import (
    RESIDUAL_DECAY,
    RESIDUAL_FREEZE,
    FullCarRiskMPC,
    build_prediction_matrices,
    discretize_full_car,
    predict_fz_horizon,
    predict_residual_horizon,
    rollout_stacked,
    rollout_step_by_step,
    zoh_discretize_with_disturbance,
)
from risk_aware_active_suspension.controllers.risk_qp import FullCarRiskAwareQP, RiskQPParams
from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.utils.config import LQRParams


def test_zoh_discretization_matches_full_car_rk4_step(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    sample_time = 0.001
    a_d, input_d = zoh_discretize_with_disturbance(
        plant.A, plant.B, plant.E, sample_time
    )
    b_d = input_d[:, :4]
    e_d = input_d[:, 4:]
    rng = np.random.default_rng(42)
    x0 = 0.01 * rng.standard_normal(14)
    u = 200.0 * rng.standard_normal(4)
    w = 0.005 * rng.standard_normal(4)

    discrete_next = a_d @ x0 + b_d @ u + e_d @ w
    rk4_next = plant.step(x0, w, sample_time, u=u)

    assert np.allclose(discrete_next, rk4_next, atol=1e-9, rtol=1e-7)


def test_prediction_stack_matches_step_by_step_rollout(default_vehicle) -> None:
    model = discretize_full_car(default_vehicle, sample_time=0.005)
    horizon = 10
    bar_a, bar_b, bar_e = model.stack(horizon)
    rng = np.random.default_rng(7)
    x0 = 0.01 * rng.standard_normal(model.n_x)
    u_seq = 150.0 * rng.standard_normal((horizon, model.n_u))
    w_seq = 0.003 * rng.standard_normal((horizon, model.n_w))

    stacked = rollout_stacked(bar_a, bar_b, bar_e, x0, u_seq, w_seq)
    stepped = rollout_step_by_step(model.A_d, model.B_d, model.E_d, x0, u_seq, w_seq)

    assert stacked.shape == (horizon, model.n_x)
    assert np.allclose(stacked, stepped, atol=1e-12, rtol=1e-12)


def test_prediction_matrix_shapes(default_vehicle) -> None:
    model = discretize_full_car(default_vehicle, sample_time=0.005)
    horizon = 5
    bar_a, bar_b, bar_e = build_prediction_matrices(
        model.A_d, model.B_d, model.E_d, horizon
    )
    assert bar_a.shape == (horizon * 14, 14)
    assert bar_b.shape == (horizon * 14, horizon * 4)
    assert bar_e.shape == (horizon * 14, horizon * 4)


def test_prediction_helpers_reject_bad_horizon(default_vehicle) -> None:
    model = discretize_full_car(default_vehicle, sample_time=0.005)
    with pytest.raises(ValueError):
        model.stack(0)


def test_horizon_one_mpc_reproduces_one_step_risk_qp_without_margin(default_vehicle) -> None:
    lqr = LQRParams()
    risk = RiskWeightParams(
        rho_th=0.75,
        k_rho=25.0,
        kappa_rho=5.0,
        q_c_min=0.8,
        q_c_max=1.0,
        q_p_min=1.0,
        q_p_max=5.0,
    )
    qp = RiskQPParams(gamma=-0.5, enforce_rate=False)
    one_step = FullCarRiskAwareQP(default_vehicle, lqr, risk, qp)
    mpc = FullCarRiskMPC(default_vehicle, lqr, risk, qp, horizon=1)
    rng = np.random.default_rng(13)
    state = 0.03 * rng.standard_normal(14)
    rho_ij = np.array([0.2, 0.8, 0.9, 0.4])

    u_qp = one_step.compute(state, rho_ij=rho_ij)
    u_mpc = mpc.compute(state, rho_ij=rho_ij)

    assert np.allclose(u_mpc, u_qp, atol=2e-3)
    assert np.allclose(mpc.last_xi, one_step.last_xi, atol=1e-6)


def test_horizon_one_mpc_reproduces_one_step_risk_qp_with_margin(default_vehicle) -> None:
    lqr = LQRParams()
    risk = RiskWeightParams(
        rho_th=0.75,
        k_rho=25.0,
        kappa_rho=5.0,
        q_c_min=0.8,
        q_c_max=1.0,
        q_p_min=1.0,
        q_p_max=5.0,
    )
    qp = RiskQPParams(gamma=-0.5, rho_safe=0.85, enforce_rate=False)
    one_step = FullCarRiskAwareQP(default_vehicle, lqr, risk, qp)
    mpc = FullCarRiskMPC(default_vehicle, lqr, risk, qp, horizon=1)
    state = np.zeros(14)
    f_z = np.array([1000.0, 4000.0, 3800.0, 3600.0])
    f_c = np.array([1200.0, 800.0, 900.0, 700.0])

    u_qp = one_step.compute(state, f_z_hat=f_z, f_c=f_c, mu=0.8)
    u_mpc = mpc.compute(state, f_z_hat=f_z, f_c=f_c, mu=0.8)

    assert np.allclose(u_mpc, u_qp, atol=2e-3)
    assert np.allclose(mpc.last_xi, one_step.last_xi, atol=2e-3)


def test_mpc_horizon_constraint_shapes_and_solution(default_vehicle) -> None:
    lqr = LQRParams()
    risk = RiskWeightParams(q_c_min=1.0, q_c_max=1.0, q_p_min=0.0, q_p_max=0.0)
    mpc = FullCarRiskMPC(
        default_vehicle,
        lqr,
        risk,
        RiskQPParams(gamma=-0.5, enforce_rate=False),
        horizon=5,
    )
    u = mpc.compute(np.zeros(14), rho_ij=np.zeros((5, 4)))
    assert u.shape == (4,)
    assert mpc.last_solution.shape == (5 * 4 + 5 * 4,)


def test_horizon_mpc_comfort_only_first_move_matches_comfort_qp(default_vehicle) -> None:
    lqr = LQRParams()
    risk = RiskWeightParams(q_c_min=1.0, q_c_max=1.0, q_p_min=0.0, q_p_max=0.0)
    qp = RiskQPParams(enforce_rate=False, r_u_factor=0.0, r_du_factor=0.0)
    mpc = FullCarRiskMPC(default_vehicle, lqr, risk, qp, horizon=10)
    comfort = FullCarComfortQP(default_vehicle, lqr)
    rng = np.random.default_rng(24)
    state = 0.01 * rng.standard_normal(14)

    u_mpc = mpc.compute(state)
    u_comfort = comfort.compute(state)

    assert np.allclose(u_mpc, u_comfort, atol=2e-3)


def test_residual_prediction_freeze_and_decay_modes() -> None:
    residual = np.array([100.0, -50.0, 25.0, 0.0])
    frozen = predict_residual_horizon(residual, horizon=4, mode=RESIDUAL_FREEZE)
    decayed = predict_residual_horizon(residual, horizon=4, mode=RESIDUAL_DECAY, alpha=0.9)

    assert np.allclose(frozen, np.tile(residual, (4, 1)))
    assert np.allclose(decayed[0], residual)
    assert np.allclose(decayed[1], 0.9 * residual)
    assert np.allclose(decayed[3], (0.9 ** 3) * residual)


def test_predict_fz_horizon_adds_predicted_residual() -> None:
    fz_bar = np.full((3, 4), 2500.0)
    residual = np.array([100.0, 0.0, -50.0, 25.0])
    fz = predict_fz_horizon(fz_bar, residual, mode=RESIDUAL_DECAY, alpha=0.5)

    assert fz.shape == (3, 4)
    assert np.allclose(fz[0], fz_bar[0] + residual)
    assert np.allclose(fz[2], fz_bar[2] + 0.25 * residual)


def test_warm_start_shifts_previous_horizon_solution(default_vehicle) -> None:
    lqr = LQRParams()
    risk = RiskWeightParams(q_c_min=1.0, q_c_max=1.0, q_p_min=0.0, q_p_max=0.0)
    mpc = FullCarRiskMPC(default_vehicle, lqr, risk, RiskQPParams(enforce_rate=False), horizon=3)
    u = np.arange(12.0).reshape(3, 4)
    xi = (100.0 + np.arange(12.0)).reshape(3, 4)
    shifted = mpc._shift_warm_start(np.concatenate([u.reshape(-1), xi.reshape(-1)]))

    assert np.allclose(shifted[:12].reshape(3, 4), np.vstack([u[1:], u[-1:]]))
    assert np.allclose(shifted[12:].reshape(3, 4), np.vstack([xi[1:], xi[-1:]]))


def test_actuator_only_warm_start_resets_slack(default_vehicle) -> None:
    lqr = LQRParams()
    risk = RiskWeightParams(q_c_min=1.0, q_c_max=1.0, q_p_min=0.0, q_p_max=0.0)
    mpc = FullCarRiskMPC(default_vehicle, lqr, risk, RiskQPParams(enforce_rate=False), horizon=3)
    u = np.arange(12.0)
    xi = 100.0 + np.arange(12.0)
    warm = mpc._actuator_only_warm_start(np.concatenate([u, xi]))

    assert np.allclose(warm[:12], u)
    assert np.allclose(warm[12:], 0.0)
