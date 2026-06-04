from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import osqp
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.signal import cont2discrete

from risk_aware_active_suspension.controllers.risk_qp import (
    RiskQPParams,
    _build_comfort_matrices,
    _resolve_gamma,
)
from risk_aware_active_suspension.controllers.risk_weights import (
    RiskWeightParams,
    compute_all as compute_risk_weights,
)
from risk_aware_active_suspension.metrics.tire import f_z_required, rho
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.utils.config import LQRParams, VehicleParams


RESIDUAL_FREEZE = "freeze"
RESIDUAL_DECAY = "decay"


@dataclass(frozen=True)
class DiscretePredictionModel:
    """Discrete full-car prediction model and finite-horizon stacks.

    The stacked convention is

        X = [x_1; x_2; ...; x_N]
        U = [u_0; u_1; ...; u_{N-1}]
        W = [w_0; w_1; ...; w_{N-1}]

    with x_{k+1} = A_d x_k + B_d u_k + E_d w_k.
    """

    A_d: np.ndarray
    B_d: np.ndarray
    E_d: np.ndarray
    G_d: np.ndarray

    @property
    def n_x(self) -> int:
        return int(self.A_d.shape[0])

    @property
    def n_u(self) -> int:
        return int(self.B_d.shape[1])

    @property
    def n_w(self) -> int:
        return int(self.E_d.shape[1])

    @property
    def n_body_acc(self) -> int:
        return int(self.G_d.shape[1])

    def stack(self, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return build_prediction_matrices(self.A_d, self.B_d, self.E_d, horizon)

    def stack_body_acc(self, horizon: int) -> np.ndarray:
        _, _, bar_g = build_prediction_matrices(self.A_d, self.B_d, self.G_d, horizon)
        return bar_g


def discretize_full_car(vehicle: VehicleParams, sample_time: float) -> DiscretePredictionModel:
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive.")
    plant = FullCar(vehicle)
    g_body = _body_acc_input_matrix(vehicle)
    a_d, input_d = zoh_discretize_with_disturbance(
        plant.A,
        plant.B,
        np.hstack([plant.E, g_body]),
        sample_time,
    )
    n_u = plant.B.shape[1]
    n_w = plant.E.shape[1]
    b_d = input_d[:, :n_u]
    e_d = input_d[:, n_u:n_u + n_w]
    g_d = input_d[:, n_u + n_w:]
    return DiscretePredictionModel(A_d=a_d, B_d=b_d, E_d=e_d, G_d=g_d)


def _body_acc_input_matrix(vehicle: VehicleParams) -> np.ndarray:
    """Continuous input matrix for known body acceleration ``[a_x, a_y]``."""
    g = np.zeros((14, 2), dtype=float)
    g[3, 1] = -vehicle.m * vehicle.h_g / vehicle.I_x
    g[5, 0] = vehicle.m * vehicle.h_g / vehicle.I_y
    return g


def zoh_discretize_with_disturbance(
    a: ArrayLike,
    b: ArrayLike,
    e: ArrayLike,
    sample_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive.")
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1]:
        raise ValueError("a must be square.")
    if b_arr.ndim != 2 or b_arr.shape[0] != a_arr.shape[0]:
        raise ValueError("b must have the same row count as a.")
    if e_arr.ndim != 2 or e_arr.shape[0] != a_arr.shape[0]:
        raise ValueError("e must have the same row count as a.")
    input_matrix = np.hstack([b_arr, e_arr])
    n_x = a_arr.shape[0]
    n_in = input_matrix.shape[1]
    system_d = cont2discrete(
        (a_arr, input_matrix, np.eye(n_x), np.zeros((n_x, n_in))),
        sample_time,
        method="zoh",
    )
    return np.asarray(system_d[0]), np.asarray(system_d[1])


def build_prediction_matrices(
    a_d: ArrayLike,
    b_d: ArrayLike,
    e_d: ArrayLike,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    a = np.asarray(a_d, dtype=float)
    b = np.asarray(b_d, dtype=float)
    e = np.asarray(e_d, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a_d must be square.")
    n_x = a.shape[0]
    if b.ndim != 2 or b.shape[0] != n_x:
        raise ValueError("b_d must have the same row count as a_d.")
    if e.ndim != 2 or e.shape[0] != n_x:
        raise ValueError("e_d must have the same row count as a_d.")
    n_u = b.shape[1]
    n_w = e.shape[1]

    powers = [np.eye(n_x)]
    for _ in range(horizon):
        powers.append(powers[-1] @ a)

    bar_a = np.zeros((horizon * n_x, n_x), dtype=float)
    bar_b = np.zeros((horizon * n_x, horizon * n_u), dtype=float)
    bar_e = np.zeros((horizon * n_x, horizon * n_w), dtype=float)
    for row_step in range(horizon):
        row = slice(row_step * n_x, (row_step + 1) * n_x)
        bar_a[row, :] = powers[row_step + 1]
        for col_step in range(row_step + 1):
            power = powers[row_step - col_step]
            u_col = slice(col_step * n_u, (col_step + 1) * n_u)
            w_col = slice(col_step * n_w, (col_step + 1) * n_w)
            bar_b[row, u_col] = power @ b
            bar_e[row, w_col] = power @ e
    return bar_a, bar_b, bar_e


def rollout_step_by_step(
    a_d: ArrayLike,
    b_d: ArrayLike,
    e_d: ArrayLike,
    x0: ArrayLike,
    u_seq: ArrayLike,
    w_seq: ArrayLike,
) -> np.ndarray:
    a = np.asarray(a_d, dtype=float)
    b = np.asarray(b_d, dtype=float)
    e = np.asarray(e_d, dtype=float)
    x = np.asarray(x0, dtype=float)
    u = np.asarray(u_seq, dtype=float)
    w = np.asarray(w_seq, dtype=float)
    if x.shape != (a.shape[0],):
        raise ValueError(f"x0 must have shape ({a.shape[0]},), got {x.shape}.")
    if u.ndim != 2 or u.shape[1] != b.shape[1]:
        raise ValueError(f"u_seq must have shape (N, {b.shape[1]}).")
    if w.ndim != 2 or w.shape != (u.shape[0], e.shape[1]):
        raise ValueError(f"w_seq must have shape ({u.shape[0]}, {e.shape[1]}).")

    states = np.zeros((u.shape[0], a.shape[0]), dtype=float)
    x_now = x.copy()
    for idx in range(u.shape[0]):
        x_now = a @ x_now + b @ u[idx] + e @ w[idx]
        states[idx] = x_now
    return states


def rollout_stacked(
    bar_a: ArrayLike,
    bar_b: ArrayLike,
    bar_e: ArrayLike,
    x0: ArrayLike,
    u_seq: ArrayLike,
    w_seq: ArrayLike,
) -> np.ndarray:
    a_bar = np.asarray(bar_a, dtype=float)
    b_bar = np.asarray(bar_b, dtype=float)
    e_bar = np.asarray(bar_e, dtype=float)
    x = np.asarray(x0, dtype=float)
    u = np.asarray(u_seq, dtype=float)
    w = np.asarray(w_seq, dtype=float)
    stacked = a_bar @ x + b_bar @ u.reshape(-1) + e_bar @ w.reshape(-1)
    return stacked.reshape(u.shape[0], x.shape[0])


def predict_residual_horizon(
    residual: ArrayLike,
    horizon: int,
    mode: str = RESIDUAL_FREEZE,
    alpha: float = 1.0,
) -> np.ndarray:
    """Predict per-corner residual force over the MPC horizon.

    ``freeze`` holds the current residual estimate constant. ``decay`` applies
    residual[k + ell] = alpha**ell * residual[k], with alpha in [0, 1].
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    res = np.asarray(residual, dtype=float)
    if res.shape != (4,):
        raise ValueError(f"residual must have shape (4,), got {res.shape}.")
    if mode == RESIDUAL_FREEZE:
        factors = np.ones(horizon, dtype=float)
    elif mode == RESIDUAL_DECAY:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1] for decay mode.")
        factors = alpha ** np.arange(horizon, dtype=float)
    else:
        raise ValueError(f"unknown residual prediction mode {mode!r}.")
    return factors[:, None] * res[None, :]


def predict_fz_horizon(
    f_z_bar_horizon: ArrayLike,
    residual: ArrayLike,
    mode: str = RESIDUAL_FREEZE,
    alpha: float = 1.0,
) -> np.ndarray:
    """Combine a nominal F_z trajectory with a predicted residual trajectory."""
    fz_bar = np.asarray(f_z_bar_horizon, dtype=float)
    if fz_bar.ndim != 2 or fz_bar.shape[1] != 4:
        raise ValueError(f"f_z_bar_horizon must have shape (N, 4), got {fz_bar.shape}.")
    return fz_bar + predict_residual_horizon(
        residual, horizon=fz_bar.shape[0], mode=mode, alpha=alpha
    )


@dataclass
class FullCarRiskMPC:
    """Finite-horizon risk-aware MPC skeleton.

    Phase 5.2 intentionally keeps the same one-step stage cost and margin
    semantics as ``FullCarRiskAwareQP``. For ``horizon=1`` this class is an
    acceptance harness: it must reproduce the Phase 4 one-step QP. For larger
    horizons it already stacks actuator, rate, nonnegative slack, and margin
    constraints; residual/road prediction modes are added in Phase 5.3.
    """

    vehicle: VehicleParams
    lqr: LQRParams
    risk: RiskWeightParams
    qp: RiskQPParams = field(default_factory=RiskQPParams)
    horizon: int = 10
    warm_start: bool = True
    reuse_solver: bool = True
    name: str = "risk_mpc_full"
    _M: np.ndarray = field(init=False, repr=False)
    _L_x: np.ndarray = field(init=False, repr=False)
    _Q_x: np.ndarray = field(init=False, repr=False)
    _M_stage: np.ndarray = field(init=False, repr=False)
    _L_x_stage: np.ndarray = field(init=False, repr=False)
    _Q_x_stage: np.ndarray = field(init=False, repr=False)
    _gamma_matrix: np.ndarray = field(init=False, repr=False)
    _prediction_model: DiscretePredictionModel = field(init=False, repr=False)
    _bar_a: np.ndarray = field(init=False, repr=False)
    _bar_b: np.ndarray = field(init=False, repr=False)
    _bar_e: np.ndarray = field(init=False, repr=False)
    _bar_g: np.ndarray = field(init=False, repr=False)
    _u_prev: np.ndarray = field(init=False, repr=False)
    _prob: osqp.OSQP | None = field(init=False, default=None, repr=False)
    _prob_p_nnz: int = field(init=False, default=0, repr=False)
    _prob_a_nnz: int = field(init=False, default=0, repr=False)
    _prob_shape: tuple[int, int, int] = field(init=False, default=(0, 0, 0), repr=False)
    last_solve_time_s: float = field(init=False, default=0.0)
    last_setup_solve_time_s: float = field(init=False, default=0.0)
    last_iterations: int = field(init=False, default=0)
    last_status: str = field(init=False, default="not_solved")
    last_sigma: float = field(init=False, default=0.0)
    last_q_c: float = field(init=False, default=1.0)
    last_q_p_local: np.ndarray = field(init=False, default_factory=lambda: np.zeros(4))
    last_xi: np.ndarray = field(init=False, default_factory=lambda: np.zeros(4))
    last_solution: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0))
    last_dual: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        comfort = _build_comfort_matrices(self.vehicle, self.lqr)
        self._M = comfort["M"]
        self._L_x = comfort["L_x"]
        self._Q_x = comfort["Q_x"]
        self._M_stage = comfort["M_stage"]
        self._L_x_stage = comfort["L_x_stage"]
        self._Q_x_stage = comfort["Q_x_stage"]
        self._gamma_matrix = _resolve_gamma(self.qp, self.vehicle, self.lqr)
        self._prediction_model = discretize_full_car(self.vehicle, self.lqr.T_s)
        self._bar_a, self._bar_b, self._bar_e = self._prediction_model.stack(self.horizon)
        self._bar_g = self._prediction_model.stack_body_acc(self.horizon)
        self._u_prev = np.zeros(4, dtype=float)

    @property
    def gamma_matrix(self) -> np.ndarray:
        return self._gamma_matrix

    @property
    def prediction_model(self) -> DiscretePredictionModel:
        return self._prediction_model

    def reset(self) -> None:
        self._u_prev[:] = 0.0
        self.last_solution = np.zeros(0)
        self.last_dual = np.zeros(0)
        self._prob = None

    def compute(
        self,
        state: ArrayLike,
        rho_ij: ArrayLike | None = None,
        f_z_hat: ArrayLike | None = None,
        f_c: ArrayLike | None = None,
        mu: float | ArrayLike | None = None,
        road_horizon: ArrayLike | None = None,
        body_acc_horizon: ArrayLike | None = None,
        f_z_residual: ArrayLike | None = None,
        *_: Any,
        **__: Any,
    ) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (14,):
            raise ValueError(f"full-car state must have shape (14,), got {x.shape}.")

        margin_enabled = f_z_hat is not None or f_c is not None or mu is not None
        fz_h = _as_horizon_array(f_z_hat, self.horizon, "f_z_hat") if f_z_hat is not None else None
        fc_h = _as_horizon_array(f_c, self.horizon, "f_c") if f_c is not None else None
        road_h = _as_horizon_array(road_horizon, self.horizon, "road_horizon") if road_horizon is not None else None
        body_acc_h = (
            _as_horizon_array(body_acc_horizon, self.horizon, "body_acc_horizon", width=2)
            if body_acc_horizon is not None
            else None
        )
        fz_residual_h = (
            _as_horizon_array(f_z_residual, self.horizon, "f_z_residual")
            if f_z_residual is not None
            else None
        )
        if margin_enabled and (fz_h is None or fc_h is None or mu is None):
            raise ValueError("f_z_hat, f_c, and mu must be supplied together for margin constraints.")
        if fz_residual_h is not None and road_h is None:
            raise ValueError("road_horizon must be supplied when f_z_residual is supplied.")

        rho_h = self._rho_horizon(rho_ij=rho_ij, fz_h=fz_h, fc_h=fc_h, mu=mu)
        q_c_h = np.zeros(self.horizon, dtype=float)
        q_p_h = np.zeros((self.horizon, 4), dtype=float)
        sigma_h = np.zeros(self.horizon, dtype=float)
        for step in range(self.horizon):
            weights = compute_risk_weights(float(np.max(rho_h[step])), rho_h[step], self.risk)
            sigma_h[step] = float(weights["sigma"])
            q_c_h[step] = float(weights["q_c"])
            q_p_h[step] = np.asarray(weights["q_p_local"], dtype=float)

        p_mat, q_vec = self._build_cost(x, q_c_h, q_p_h)
        a_mat, l_vec, u_vec = self._build_constraints(
            x0=x,
            fz_h=fz_h,
            fc_h=fc_h,
            mu=mu,
            road_h=road_h,
            body_acc_h=body_acc_h,
            fz_residual_h=fz_residual_h,
        )

        t_setup = time.perf_counter()
        problem = self._prepare_problem(p_mat, q_vec, a_mat, l_vec, u_vec)
        if self.warm_start and self.last_solution.shape == (8 * self.horizon,):
            # Receding-horizon warm start: shift the previous primal solution
            # by one stage (repeating the terminal stage) and reuse the
            # previous dual, so OSQP retains the active-set information that
            # accelerates ADMM convergence between consecutive solves.
            y_ws = self.last_dual if self.last_dual.shape == (a_mat.shape[0],) else None
            problem.warm_start(x=self._shift_warm_start(self.last_solution), y=y_ws)
        elif not self.warm_start:
            problem.warm_start(x=np.zeros(8 * self.horizon), y=np.zeros(a_mat.shape[0]))
        t0 = time.perf_counter()
        res = problem.solve(raise_error=False)
        self.last_solve_time_s = time.perf_counter() - t0
        self.last_setup_solve_time_s = time.perf_counter() - t_setup
        self.last_status = str(getattr(res.info, "status", "unknown"))
        self.last_iterations = int(getattr(res.info, "iter", 0))
        if self.last_status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"Risk-MPC OSQP failed: status={self.last_status}")

        z = np.asarray(res.x, dtype=float)
        n_u_total = 4 * self.horizon
        u0 = z[:4].copy()
        xi0 = z[n_u_total:n_u_total + 4].copy()
        if margin_enabled:
            xi0[np.abs(xi0) < 1.0e-5] = 0.0
        else:
            xi0 = np.zeros(4)
        self.last_sigma = float(sigma_h[0])
        self.last_q_c = float(q_c_h[0])
        self.last_q_p_local = q_p_h[0].copy()
        self.last_xi = xi0
        self.last_solution = z
        self.last_dual = np.asarray(getattr(res, "y", np.zeros(0)), dtype=float)
        self._u_prev = u0.copy()
        return u0

    def _prepare_problem(
        self,
        p_mat: sp.csc_matrix,
        q_vec: np.ndarray,
        a_mat: sp.csc_matrix,
        l_vec: np.ndarray,
        u_vec: np.ndarray,
    ) -> osqp.OSQP:
        shape = (p_mat.shape[0], a_mat.shape[0], a_mat.shape[1])
        can_reuse = (
            self.reuse_solver
            and self._prob is not None
            and self._prob_p_nnz == p_mat.nnz
            and self._prob_a_nnz == a_mat.nnz
            and self._prob_shape == shape
        )
        if can_reuse:
            assert self._prob is not None
            self._prob.update(Px=p_mat.data, q=q_vec, l=l_vec, u=u_vec)
            return self._prob

        problem = osqp.OSQP()
        problem.setup(
            P=p_mat,
            q=q_vec,
            A=a_mat,
            l=l_vec,
            u=u_vec,
            verbose=False,
            warm_starting=True,
            polishing=False,
            eps_abs=self.qp.osqp_eps_abs,
            eps_rel=self.qp.osqp_eps_rel,
            max_iter=self.qp.osqp_max_iter,
        )
        if self.reuse_solver:
            self._prob = problem
            self._prob_p_nnz = p_mat.nnz
            self._prob_a_nnz = a_mat.nnz
            self._prob_shape = shape
        return problem

    def _shift_warm_start(self, z_prev: np.ndarray) -> np.ndarray:
        n_u_total = 4 * self.horizon
        u_prev = z_prev[:n_u_total].reshape(self.horizon, 4)
        xi_prev = z_prev[n_u_total:].reshape(self.horizon, 4)
        u_shift = np.vstack([u_prev[1:], u_prev[-1:]])
        xi_shift = np.vstack([xi_prev[1:], xi_prev[-1:]])
        return np.concatenate([u_shift.reshape(-1), xi_shift.reshape(-1)])

    def _actuator_only_warm_start(self, z_prev: np.ndarray) -> np.ndarray:
        n_u_total = 4 * self.horizon
        out = np.zeros(8 * self.horizon, dtype=float)
        out[:n_u_total] = z_prev[:n_u_total]
        return out

    def _rho_horizon(
        self,
        rho_ij: ArrayLike | None,
        fz_h: np.ndarray | None,
        fc_h: np.ndarray | None,
        mu: float | ArrayLike | None,
    ) -> np.ndarray:
        if rho_ij is not None:
            return _as_horizon_array(rho_ij, self.horizon, "rho_ij")
        if fz_h is not None and fc_h is not None and mu is not None:
            return np.asarray(rho(fc_h, 0.0, fz_h, mu), dtype=float)
        return np.zeros((self.horizon, 4), dtype=float)

    def _build_cost(
        self,
        x0: np.ndarray,
        q_c_h: np.ndarray,
        q_p_h: np.ndarray,
    ) -> tuple[sp.csc_matrix, np.ndarray]:
        n_u_total = 4 * self.horizon
        n_xi_total = 4 * self.horizon
        blocks_xi = []
        h_u = np.zeros((n_u_total, n_u_total), dtype=float)
        q_u = np.zeros(n_u_total, dtype=float)
        for step in range(self.horizon):
            selector = np.zeros((4, n_u_total), dtype=float)
            selector[:, 4 * step:4 * (step + 1)] = np.eye(4)
            if step == 0:
                s_x = np.zeros((14, n_u_total), dtype=float)
                c_x = x0
            else:
                row = slice((step - 1) * 14, step * 14)
                s_x = self._bar_b[row, :]
                c_x = self._bar_a[row, :] @ x0

            q_c_step = q_c_h[step]
            # Use raw stage matrices on intermediate steps; absorb the terminal
            # x_N^T P x_N cost into step N-1 by switching to the augmented M, L_x, Q_x.
            if step == self.horizon - 1:
                M_k = self._M
                Lx_k = self._L_x
                Qx_k = self._Q_x
            else:
                M_k = self._M_stage
                Lx_k = self._L_x_stage
                Qx_k = self._Q_x_stage
            h_u += 2.0 * q_c_step * (
                selector.T @ M_k @ selector
                + s_x.T @ Qx_k @ s_x
                + selector.T @ Lx_k @ s_x
                + s_x.T @ Lx_k.T @ selector
            )
            h_u += 2.0 * self.qp.r_u_factor * (selector.T @ selector)
            if self.qp.r_du_factor > 0.0:
                if step == 0:
                    h_u += 2.0 * self.qp.r_du_factor * (selector.T @ selector)
                    q_u += -2.0 * self.qp.r_du_factor * (selector.T @ self._u_prev)
                else:
                    diff = np.zeros((4, n_u_total), dtype=float)
                    diff[:, 4 * step:4 * (step + 1)] = np.eye(4)
                    diff[:, 4 * (step - 1):4 * step] = -np.eye(4)
                    h_u += 2.0 * self.qp.r_du_factor * (diff.T @ diff)

            q_u += 2.0 * q_c_step * (
                s_x.T @ Qx_k @ c_x
                + selector.T @ Lx_k @ c_x
            )
            h_xi = np.diag(2.0 * q_p_h[step] + self.qp.xi_regularization)
            blocks_xi.append(sp.csc_matrix(h_xi))

        p = sp.block_diag([sp.csc_matrix(h_u)] + blocks_xi, format="csc")
        return sp.triu(p, format="csc"), np.concatenate([q_u, np.zeros(n_xi_total)])

    def _build_constraints(
        self,
        x0: np.ndarray,
        fz_h: np.ndarray | None,
        fc_h: np.ndarray | None,
        mu: float | ArrayLike | None,
        road_h: np.ndarray | None,
        body_acc_h: np.ndarray | None,
        fz_residual_h: np.ndarray | None,
    ) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
        h = self.horizon
        n_u_total = 4 * h
        n_xi_total = 4 * h
        n_z = n_u_total + n_xi_total
        f_max = self.lqr.f_max
        rate_amp = self.qp.df_max * self.lqr.T_s if self.qp.enforce_rate else 1.0e12

        rows = []
        lower = []
        upper = []

        # Actuator bounds.
        rows.append(sp.hstack([sp.eye(n_u_total), sp.csc_matrix((n_u_total, n_xi_total))]))
        lower.append(-f_max * np.ones(n_u_total))
        upper.append(f_max * np.ones(n_u_total))

        # Slack bounds.
        rows.append(sp.hstack([sp.csc_matrix((n_xi_total, n_u_total)), sp.eye(n_xi_total)]))
        lower.append(np.zeros(n_xi_total))
        upper.append(1.0e12 * np.ones(n_xi_total))

        # Rate bounds: u_0 - u_prev, then u_l - u_{l-1}.
        d_rate = sp.lil_matrix((n_u_total, n_u_total))
        for step in range(h):
            row = slice(4 * step, 4 * (step + 1))
            col = slice(4 * step, 4 * (step + 1))
            d_rate[row, col] = sp.eye(4)
            if step > 0:
                prev = slice(4 * (step - 1), 4 * step)
                d_rate[row, prev] = -sp.eye(4)
        rows.append(sp.hstack([d_rate.tocsc(), sp.csc_matrix((n_u_total, n_xi_total))]))
        rate_lower = -rate_amp * np.ones(n_u_total)
        rate_upper = rate_amp * np.ones(n_u_total)
        rate_lower[:4] += self._u_prev
        rate_upper[:4] += self._u_prev
        lower.append(rate_lower)
        upper.append(rate_upper)

        # Margin rows. Without a road forecast this preserves the Phase 4
        # gamma-margin semantics. With road_horizon, Phase 5 constrains the
        # horizon-predicted physical tire load, not a one-step gamma surrogate.
        if road_h is None:
            gamma_blocks = [sp.csc_matrix(self._gamma_matrix) for _ in range(h)]
            margin_u = sp.block_diag(gamma_blocks, format="csc")
            rows.append(sp.hstack([margin_u, sp.eye(n_xi_total)]))
            lower.append(self._margin_lower(fz_h=fz_h, fc_h=fc_h, mu=mu))
        else:
            fz_u, fz_const = self._physical_fz_prediction_affine(
                x0=x0,
                road_h=road_h,
                body_acc_h=body_acc_h,
                fz_residual_h=fz_residual_h,
            )
            rows.append(sp.hstack([sp.csc_matrix(fz_u), sp.eye(n_xi_total)]))
            lower.append(
                (np.asarray(f_z_required(fc_h, mu, self.qp.rho_safe), dtype=float).reshape(-1) - fz_const)
                if fc_h is not None and mu is not None
                else -1.0e12 * np.ones(n_xi_total)
            )
        upper.append(1.0e12 * np.ones(n_xi_total))

        a_mat = sp.vstack(rows, format="csc")
        if a_mat.shape[1] != n_z:
            raise RuntimeError("internal MPC constraint width mismatch.")
        return a_mat, np.concatenate(lower), np.concatenate(upper)

    def _margin_lower(
        self,
        fz_h: np.ndarray | None,
        fc_h: np.ndarray | None,
        mu: float | ArrayLike | None,
    ) -> np.ndarray:
        if fz_h is None and fc_h is None and mu is None:
            return -1.0e12 * np.ones(4 * self.horizon)
        if fz_h is None or fc_h is None or mu is None:
            raise ValueError("f_z_hat, f_c, and mu must be supplied together for margin constraints.")
        return (f_z_required(fc_h, mu, self.qp.rho_safe) - fz_h).reshape(-1)

    def _physical_fz_prediction_affine(
        self,
        x0: np.ndarray,
        road_h: np.ndarray,
        body_acc_h: np.ndarray | None,
        fz_residual_h: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``Fz_pred = fz_u @ U + fz_const`` over the horizon."""
        h = self.horizon
        n_u_total = 4 * h
        road_flat = road_h.reshape(-1)
        residual = np.zeros((h, 4), dtype=float) if fz_residual_h is None else fz_residual_h
        fz_u = np.zeros((4 * h, n_u_total), dtype=float)
        fz_const = np.zeros(4 * h, dtype=float)
        static = FullCar(self.vehicle).static_loads()
        tire_rows = [6, 8, 10, 12]
        for step in range(h):
            x_row = slice(step * 14, (step + 1) * 14)
            out_row = slice(step * 4, (step + 1) * 4)
            x_const = self._bar_a[x_row, :] @ x0 + self._bar_e[x_row, :] @ road_flat
            if body_acc_h is not None:
                x_const = x_const + self._bar_g[x_row, :] @ body_acc_h.reshape(-1)
            x_u = self._bar_b[x_row, :]
            for corner, state_row in enumerate(tire_rows):
                row = out_row.start + corner
                fz_u[row, :] = -self.vehicle.k_t * x_u[state_row, :]
                fz_const[row] = (
                    static[corner]
                    + self.vehicle.k_t * (road_h[step, corner] - x_const[state_row])
                    + residual[step, corner]
                )
        return fz_u, fz_const


def _as_horizon_array(value: ArrayLike, horizon: int, name: str, width: int = 4) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (width,):
        return np.tile(arr, (horizon, 1))
    if arr.shape == (horizon, width):
        return arr
    raise ValueError(f"{name} must have shape ({width},) or ({horizon}, {width}), got {arr.shape}.")
