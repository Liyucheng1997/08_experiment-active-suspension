from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_are

from risk_aware_active_suspension.controllers.lqr import _zoh_discretize
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.solvers.qp_harness import BoxQP, QPSolveResult
from risk_aware_active_suspension.utils.config import LQRParams, VehicleParams


def _build_comfort_qp_matrices(
    a_d: np.ndarray,
    b_d: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    q_y: np.ndarray,
    r: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the comfort-QP cost so its unconstrained minimum equals LQR.

    Cost form  min ||G_a u + a_0||^2_{Q_a} + u^T R_u u  with
        G_a = [D; B_d],  a_0 = [C; A_d] x,  Q_a = blkdiag(Q_y, P),  R_u = R,
    where P solves the discrete ARE for (A_d, B_d) with output weights (Q_y, R).
    Expanding gives the OSQP-form Hessian H_osqp = 2 (G_a^T Q_a G_a + R_u)
    and linear term q(x) = 2 (G_a^T Q_a [C; A_d]) x.
    """
    q_state = c.T @ q_y @ c
    s = c.T @ q_y @ d
    r_eff = r + d.T @ q_y @ d
    p = solve_discrete_are(a_d, b_d, q_state, r_eff, s=s)

    h = r_eff + b_d.T @ p @ b_d  # Hessian / 2 in OSQP convention
    linear_gain = s.T + b_d.T @ p @ a_d  # f^T x; per OSQP: q = 2 * linear_gain @ x
    k_lqr = np.linalg.solve(h, linear_gain)
    return {"H": h, "linear_gain": linear_gain, "P_dare": p, "K_lqr": k_lqr}


@dataclass
class QuarterCarComfortQP:
    params: VehicleParams
    lqr: LQRParams
    name: str = "comfort_qp_quarter"
    _qp: BoxQP = field(init=False, repr=False)
    _linear_gain: np.ndarray = field(init=False, repr=False)
    _k_lqr: np.ndarray = field(init=False, repr=False)
    last_solve_time_s: float = field(init=False, default=0.0)
    last_iterations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        plant = QuarterCar(self.params)
        a_d, b_d = _zoh_discretize(plant.A, plant.B, self.lqr.T_s)
        c = plant.A[1:2, :]
        d = plant.B[1:2, :]
        q_y = np.array([[self.lqr.q_accel]], dtype=float)
        r = np.array([[self.lqr.r_force]], dtype=float)
        m = _build_comfort_qp_matrices(a_d, b_d, c, d, q_y, r)
        f_max = self.lqr.f_max
        qp = BoxQP(
            hessian=2.0 * m["H"],
            lower=np.array([-f_max], dtype=float),
            upper=np.array([f_max], dtype=float),
        )
        self._qp = qp
        self._linear_gain = m["linear_gain"]
        self._k_lqr = m["K_lqr"]

    @property
    def k_lqr(self) -> np.ndarray:
        return self._k_lqr

    def reset(self) -> None:
        return None

    def compute(self, state: ArrayLike, *_: Any, **__: Any) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (4,):
            raise ValueError(f"quarter-car state must have shape (4,), got {x.shape}.")
        q_lin = 2.0 * (self._linear_gain @ x)
        self._qp.update_linear(q_lin)
        res: QPSolveResult = self._qp.solve()
        self.last_solve_time_s = res.solve_time_s
        self.last_iterations = res.iterations
        return res.u


@dataclass
class FullCarComfortQP:
    params: VehicleParams
    lqr: LQRParams
    name: str = "comfort_qp_full"
    _qp: BoxQP = field(init=False, repr=False)
    _linear_gain: np.ndarray = field(init=False, repr=False)
    _k_lqr: np.ndarray = field(init=False, repr=False)
    last_solve_time_s: float = field(init=False, default=0.0)
    last_iterations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        plant = FullCar(self.params)
        a_d, b_d = _zoh_discretize(plant.A, plant.B, self.lqr.T_s)
        c = np.zeros((5, 14), dtype=float)
        d = np.zeros((5, 4), dtype=float)
        c[0, :] = plant.A[1, :]
        d[0, :] = plant.B[1, :]
        c[1, 2] = 1.0
        c[2, 3] = 1.0
        c[3, 4] = 1.0
        c[4, 5] = 1.0
        q_y = np.diag(
            [
                self.lqr.q_accel,
                self.lqr.q_phi,
                self.lqr.q_dphi,
                self.lqr.q_theta,
                self.lqr.q_dtheta,
            ]
        )
        r = self.lqr.r_force * np.eye(4)
        m = _build_comfort_qp_matrices(a_d, b_d, c, d, q_y, r)
        f_max = self.lqr.f_max
        qp = BoxQP(
            hessian=2.0 * m["H"],
            lower=-f_max * np.ones(4, dtype=float),
            upper=f_max * np.ones(4, dtype=float),
        )
        self._qp = qp
        self._linear_gain = m["linear_gain"]
        self._k_lqr = m["K_lqr"]

    @property
    def k_lqr(self) -> np.ndarray:
        return self._k_lqr

    def reset(self) -> None:
        return None

    def compute(self, state: ArrayLike, *_: Any, **__: Any) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (14,):
            raise ValueError(f"full-car state must have shape (14,), got {x.shape}.")
        q_lin = 2.0 * (self._linear_gain @ x)
        self._qp.update_linear(q_lin)
        res: QPSolveResult = self._qp.solve()
        self.last_solve_time_s = res.solve_time_s
        self.last_iterations = res.iterations
        return res.u
