from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import osqp
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_are

from risk_aware_active_suspension.controllers.lqr import _zoh_discretize
from risk_aware_active_suspension.controllers.risk_weights import (
    RiskWeightParams,
    compute_all as compute_risk_weights,
)
from risk_aware_active_suspension.metrics.tire import f_z_required, rho
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.utils.config import LQRParams, VehicleParams


@dataclass(frozen=True)
class RiskQPParams:
    """Knobs specific to the risk-aware QP (rate constraint, regularization).

    The margin coefficient γ — relating per-corner actuator force to its short-
    term shift on tire normal load — can be supplied three ways, in priority:
        1. ``gamma_matrix`` (4×4) used verbatim;
        2. ``gamma`` (scalar) expanded to γ·I_4 (diagonal-only, for backward
           compatibility with Phase 4.4-style tests);
        3. neither supplied → derived from the discretized full-car plant via
           the cumulative L-step response (see ``build_gamma_from_plant``).
    """

    df_max: float = 20000.0
    r_u_factor: float = 0.0
    r_du_factor: float = 0.0
    enforce_rate: bool = True
    xi_regularization: float = 1.0e-6
    osqp_eps_abs: float = 1.0e-8
    osqp_eps_rel: float = 1.0e-8
    osqp_max_iter: int = 4000
    gamma: float | None = None
    gamma_matrix: np.ndarray | None = None
    gamma_n_horizon: int = 10
    rho_safe: float = 0.85

    def __post_init__(self) -> None:
        if self.df_max <= 0.0:
            raise ValueError("df_max must be positive.")
        if self.r_u_factor < 0.0 or self.r_du_factor < 0.0:
            raise ValueError("r_u_factor and r_du_factor must be non-negative.")
        if self.xi_regularization <= 0.0:
            raise ValueError("xi_regularization must be positive.")
        if not (0.0 < self.rho_safe <= 1.0):
            raise ValueError("rho_safe must be in (0, 1].")
        if self.gamma_n_horizon <= 0:
            raise ValueError("gamma_n_horizon must be positive.")
        if self.gamma_matrix is not None:
            arr = np.asarray(self.gamma_matrix, dtype=float)
            if arr.shape != (4, 4):
                raise ValueError(f"gamma_matrix must have shape (4, 4), got {arr.shape}.")


def build_gamma_from_plant(
    vehicle: VehicleParams, lqr: LQRParams, n_horizon: int = 10
) -> np.ndarray:
    """Per-corner γ_ij = -k_t · row_{6+2i}((Σ_{ℓ=0}^{L-1} A_d^ℓ) B_d).

    Interpretation: if a constant per-corner actuator force vector u is applied
    for L control steps starting from the current state, the resulting shift in
    z_u_i is row_{6+2i}(S B_d) · u, and ΔF_z_i ≈ -k_t · Δz_u_i. The matrix is
    the controller's "look-ahead authority" over each tire load.

    For T_s = 5 ms and L = 10 the diagonal is ~+1.0 and off-diagonal ~0.05;
    deeper L lets the underdamped suspension oscillate the sign back, so the
    default L = 10 captures the transient peak relevant for ~30 ms road bumps.
    """
    if n_horizon <= 0:
        raise ValueError("n_horizon must be positive.")
    plant = FullCar(vehicle)
    a_d, b_d = _zoh_discretize(plant.A, plant.B, lqr.T_s)
    n = a_d.shape[0]
    sumA = np.zeros_like(a_d)
    Apow = np.eye(n)
    for _ in range(int(n_horizon)):
        sumA = sumA + Apow
        Apow = Apow @ a_d
    response = sumA @ b_d
    rows = [6, 8, 10, 12]
    return -vehicle.k_t * response[rows, :]


def _resolve_gamma(qp: "RiskQPParams", vehicle: VehicleParams, lqr: LQRParams) -> np.ndarray:
    if qp.gamma_matrix is not None:
        return np.asarray(qp.gamma_matrix, dtype=float)
    if qp.gamma is not None:
        return float(qp.gamma) * np.eye(4)
    return build_gamma_from_plant(vehicle, lqr, n_horizon=qp.gamma_n_horizon)


def _build_comfort_matrices(
    vehicle: VehicleParams, lqr: LQRParams
) -> dict[str, np.ndarray]:
    """Same comfort cost as Phase 3.4's _build_comfort_qp_matrices.

    Returns M = R_eff + B_d^T P B_d (so that ||G_a u + a_0||^2_{Q_a} = u^T M u + 2 u^T L_x x + const)
    and L_x = S^T + B_d^T P A_d. With q_c = 1, q_p = 0, R_u = R_du = 0, and
    rate bounds slack, the risk-QP collapses to comfort-QP / LQR.
    """
    plant = FullCar(vehicle)
    a_d, b_d = _zoh_discretize(plant.A, plant.B, lqr.T_s)
    c = np.zeros((5, 14), dtype=float)
    d = np.zeros((5, 4), dtype=float)
    c[0, :] = plant.A[1, :]
    d[0, :] = plant.B[1, :]
    c[1, 2] = 1.0
    c[2, 3] = 1.0
    c[3, 4] = 1.0
    c[4, 5] = 1.0
    q_y = np.diag([lqr.q_accel, lqr.q_phi, lqr.q_dphi, lqr.q_theta, lqr.q_dtheta])
    r = lqr.r_force * np.eye(4)
    q_state = c.T @ q_y @ c
    s_cross = c.T @ q_y @ d
    r_eff = r + d.T @ q_y @ d
    p_dare = solve_discrete_are(a_d, b_d, q_state, r_eff, s=s_cross)
    q_x = q_state + a_d.T @ p_dare @ a_d
    return {
        # Augmented matrices: stage cost ⊕ terminal cost. Use ONLY at step N-1 of
        # an MPC horizon (where ‖x_N‖²_P collapses into the stage cost at N-1)
        # — applying them at every stage double-counts P_dare.
        "M": r_eff + b_d.T @ p_dare @ b_d,
        "L_x": s_cross.T + b_d.T @ p_dare @ a_d,
        "Q_x": q_x,
        # Raw stage matrices (no terminal). Use at steps 0..N-2 of the MPC horizon.
        "M_stage": r_eff,
        "L_x_stage": s_cross.T,
        "Q_x_stage": q_state,
        "P_dare": p_dare,
        "T_s": lqr.T_s,
    }


@dataclass
class FullCarRiskAwareQP:
    """Risk-aware QP controller for the full car (Phase 4.3 — without margin constraint).

    Decision z = [u_s (4); xi (4)], cost
        J = q_c(σ) · ||G_a u + a_0||^2_{Q_a}    (comfort, time-varying)
          + Σ q_{p,ij}(σ, ρ_ij) · xi_ij^2       (slack penalty, time-varying)
          + u^T R_u u                            (optional)
          + (u - u_prev)^T R_du (u - u_prev)     (optional)

    Constraints: |u_s| ≤ f_max ; xi ≥ 0 ; |u_s - u_prev| ≤ df_max·T_s when enforced.
    If F_z/F_c/mu are supplied, also enforces
        F_z_hat,ij + gamma·F_e,ij + xi_ij ≥ F_z_required,ij.
    Otherwise the margin rows are inactive and xi* = 0 by construction.
    """

    vehicle: VehicleParams
    lqr: LQRParams
    risk: RiskWeightParams
    qp: RiskQPParams = field(default_factory=RiskQPParams)
    name: str = "risk_qp_full"
    _M: np.ndarray = field(init=False, repr=False)
    _L_x: np.ndarray = field(init=False, repr=False)
    _gamma_matrix: np.ndarray = field(init=False, repr=False)
    _prob: osqp.OSQP = field(init=False, repr=False)
    _u_prev: np.ndarray = field(init=False, repr=False)
    _Px_idx: np.ndarray = field(init=False, repr=False)
    _P_template: sp.csc_matrix = field(init=False, repr=False)
    last_solve_time_s: float = field(init=False, default=0.0)
    last_sigma: float = field(init=False, default=0.0)
    last_q_c: float = field(init=False, default=1.0)
    last_q_p_local: np.ndarray = field(init=False, default_factory=lambda: np.zeros(4))
    last_xi: np.ndarray = field(init=False, default_factory=lambda: np.zeros(4))
    last_status: str = field(init=False, default="not_solved")

    def __post_init__(self) -> None:
        comfort = _build_comfort_matrices(self.vehicle, self.lqr)
        object.__setattr__(self, "_M", comfort["M"])
        object.__setattr__(self, "_L_x", comfort["L_x"])
        gamma_matrix = _resolve_gamma(self.qp, self.vehicle, self.lqr)
        object.__setattr__(self, "_gamma_matrix", gamma_matrix)

        # Initial Hessian uses σ = 0 ⇒ q_c = q_c_max, q_p_local = q_p_min.
        H_uu0 = 2.0 * (
            self.risk.q_c_max * self._M
            + self.qp.r_u_factor * np.eye(4)
            + self.qp.r_du_factor * np.eye(4)
        )
        H_xixi0 = 2.0 * self.risk.q_p_min * np.eye(4) + self.qp.xi_regularization * np.eye(4)
        H_full = np.block(
            [
                [H_uu0, np.zeros((4, 4))],
                [np.zeros((4, 4)), H_xixi0],
            ]
        )
        # OSQP requires upper-triangular sparse Hessian.
        P_template = sp.csc_matrix(np.triu(H_full))
        object.__setattr__(self, "_P_template", P_template.copy())
        object.__setattr__(self, "_Px_idx", np.arange(P_template.nnz))

        # Constraint matrix: [box_u ; box_xi ; rate_u ; margin].
        A_cons = sp.vstack(
            [
                sp.hstack([sp.eye(4), sp.csc_matrix((4, 4))]),  # box on u
                sp.hstack([sp.csc_matrix((4, 4)), sp.eye(4)]),  # box on xi
                sp.hstack([sp.eye(4), sp.csc_matrix((4, 4))]),  # rate on u
                sp.hstack(
                    [
                        sp.csc_matrix(self._gamma_matrix),
                        sp.eye(4),
                    ]
                ),  # margin: gamma_matrix @ u + xi >= Fz_req - Fz
            ],
            format="csc",
        )
        f_max = self.lqr.f_max
        rate_amp = self.qp.df_max * self.lqr.T_s if self.qp.enforce_rate else 1.0e12
        l_init = np.concatenate(
            [
                -f_max * np.ones(4),
                np.zeros(4),
                -rate_amp * np.ones(4),
                -1.0e12 * np.ones(4),
            ]
        )
        u_init = np.concatenate(
            [
                f_max * np.ones(4),
                1.0e12 * np.ones(4),
                rate_amp * np.ones(4),
                1.0e12 * np.ones(4),
            ]
        )

        prob = osqp.OSQP()
        prob.setup(
            P=P_template,
            q=np.zeros(8),
            A=A_cons,
            l=l_init,
            u=u_init,
            verbose=False,
            warm_starting=True,
            polishing=False,
            eps_abs=self.qp.osqp_eps_abs,
            eps_rel=self.qp.osqp_eps_rel,
            max_iter=self.qp.osqp_max_iter,
        )
        object.__setattr__(self, "_prob", prob)
        object.__setattr__(self, "_u_prev", np.zeros(4))

    def reset(self) -> None:
        self._u_prev[:] = 0.0

    @property
    def comfort_matrices(self) -> dict[str, np.ndarray]:
        return {"M": self._M, "L_x": self._L_x}

    @property
    def gamma_matrix(self) -> np.ndarray:
        return self._gamma_matrix

    def compute(
        self,
        state: ArrayLike,
        rho_ij: ArrayLike | None = None,
        f_z_hat: ArrayLike | None = None,
        f_c: ArrayLike | None = None,
        mu: float | ArrayLike | None = None,
        *_: Any,
        **__: Any,
    ) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (14,):
            raise ValueError(f"full-car state must have shape (14,), got {x.shape}.")
        margin_enabled = f_z_hat is not None or f_c is not None or mu is not None
        if rho_ij is None and f_z_hat is not None and f_c is not None and mu is not None:
            rho_arr = np.asarray(rho(f_c, 0.0, f_z_hat, mu), dtype=float)
        else:
            rho_arr = np.zeros(4, dtype=float) if rho_ij is None else np.asarray(rho_ij, dtype=float)
        if rho_arr.shape != (4,):
            raise ValueError(f"rho_ij must have shape (4,), got {rho_arr.shape}.")

        weights = compute_risk_weights(float(np.max(rho_arr)), rho_arr, self.risk)
        sigma_val = float(weights["sigma"])
        q_c_val = float(weights["q_c"])
        q_p_local = np.asarray(weights["q_p_local"], dtype=float)

        # Assemble Hessian (upper triangular CSC, same sparsity as template).
        H_uu = 2.0 * (
            q_c_val * self._M
            + self.qp.r_u_factor * np.eye(4)
            + self.qp.r_du_factor * np.eye(4)
        )
        H_xixi = np.diag(2.0 * q_p_local + self.qp.xi_regularization)
        H_full = np.block(
            [
                [H_uu, np.zeros((4, 4))],
                [np.zeros((4, 4)), H_xixi],
            ]
        )
        P_new = sp.csc_matrix(np.triu(H_full))
        # Sparsity must match the template — safety check while developing.
        if P_new.nnz != self._P_template.nnz:
            raise RuntimeError("Hessian sparsity pattern changed unexpectedly.")

        # Linear: q_u = 2 q_c L_x x − 2 r_du u_prev ; q_xi = 0
        q_u = 2.0 * q_c_val * (self._L_x @ x) - 2.0 * self.qp.r_du_factor * self._u_prev
        q_lin = np.concatenate([q_u, np.zeros(4)])

        # Rate bounds (only the last 4 of l/u change per step).
        f_max = self.lqr.f_max
        rate_amp = self.qp.df_max * self.lqr.T_s if self.qp.enforce_rate else 1.0e12
        l_new = np.concatenate(
            [
                -f_max * np.ones(4),
                np.zeros(4),
                self._u_prev - rate_amp,
                self._margin_lower(f_z_hat=f_z_hat, f_c=f_c, mu=mu),
            ]
        )
        u_new = np.concatenate(
            [
                f_max * np.ones(4),
                1.0e12 * np.ones(4),
                self._u_prev + rate_amp,
                1.0e12 * np.ones(4),
            ]
        )

        self._prob.update(Px=P_new.data, q=q_lin, l=l_new, u=u_new)
        t0 = time.perf_counter()
        res = self._prob.solve(raise_error=False)
        self.last_solve_time_s = time.perf_counter() - t0
        status = str(getattr(res.info, "status", "unknown"))
        self.last_status = status
        if status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"Risk-QP OSQP failed: status={status}")

        z = np.asarray(res.x, dtype=float)
        u_s = z[:4]
        xi = z[4:]
        if margin_enabled:
            xi[np.abs(xi) < 1.0e-5] = 0.0
        else:
            xi = np.zeros(4)
        self.last_sigma = sigma_val
        self.last_q_c = q_c_val
        self.last_q_p_local = q_p_local
        self.last_xi = xi
        self._u_prev = u_s.copy()
        return u_s

    def _margin_lower(
        self,
        f_z_hat: ArrayLike | None,
        f_c: ArrayLike | None,
        mu: float | ArrayLike | None,
    ) -> np.ndarray:
        if f_z_hat is None and f_c is None and mu is None:
            return -1.0e12 * np.ones(4)
        if f_z_hat is None or f_c is None or mu is None:
            raise ValueError("f_z_hat, f_c, and mu must be supplied together for margin constraints.")
        fz = np.asarray(f_z_hat, dtype=float)
        fc = np.asarray(f_c, dtype=float)
        if fz.shape != (4,):
            raise ValueError(f"f_z_hat must have shape (4,), got {fz.shape}.")
        if fc.shape != (4,):
            raise ValueError(f"f_c must have shape (4,), got {fc.shape}.")
        return np.asarray(f_z_required(fc, mu, self.qp.rho_safe), dtype=float) - fz
