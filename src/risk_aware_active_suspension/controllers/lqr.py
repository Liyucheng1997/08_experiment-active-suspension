from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.utils.config import LQRParams, VehicleParams


def _zoh_discretize(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = a.shape[0], b.shape[1]
    sys_d = cont2discrete((a, b, np.eye(n), np.zeros((n, m))), dt, method="zoh")
    return np.asarray(sys_d[0]), np.asarray(sys_d[1])


def _lqr_gain(
    a_d: np.ndarray,
    b_d: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    q_y: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete LQR with output-weighted cost J = Σ y^T Q_y y + u^T R u.

    Expands y = C x + D u, giving the standard form
        Q = C^T Q_y C, S = C^T Q_y D, R_eff = R + D^T Q_y D.
    """
    q = c.T @ q_y @ c
    s = c.T @ q_y @ d
    r_eff = r + d.T @ q_y @ d
    p = solve_discrete_are(a_d, b_d, q, r_eff, s=s)
    k = np.linalg.solve(r_eff + b_d.T @ p @ b_d, b_d.T @ p @ a_d + s.T)
    return k, p


@dataclass(frozen=True)
class QuarterCarLQRController:
    params: VehicleParams
    lqr: LQRParams
    name: str = "lqr_quarter"
    _gain: np.ndarray = field(init=False, repr=False)
    _closed_loop_poles: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        plant = QuarterCar(self.params)
        a_d, b_d = _zoh_discretize(plant.A, plant.B, self.lqr.T_s)
        # Output: ddot z_s = A[1,:] x + B[1,:] u (use continuous A,B for output map)
        c = plant.A[1:2, :]
        d = plant.B[1:2, :]
        q_y = np.array([[self.lqr.q_accel]], dtype=float)
        r = np.array([[self.lqr.r_force]], dtype=float)
        gain, _ = _lqr_gain(a_d, b_d, c, d, q_y, r)
        poles = np.linalg.eigvals(a_d - b_d @ gain)
        object.__setattr__(self, "_gain", gain)
        object.__setattr__(self, "_closed_loop_poles", poles)

    @property
    def gain(self) -> np.ndarray:
        return self._gain

    @property
    def closed_loop_poles(self) -> np.ndarray:
        return self._closed_loop_poles

    def reset(self) -> None:
        return None

    def compute(self, state: ArrayLike, *_: Any, **__: Any) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (4,):
            raise ValueError(f"quarter-car state must have shape (4,), got {x.shape}.")
        force = float(-(self._gain @ x)[0])
        if self.lqr.f_max is not None:
            force = float(np.clip(force, -self.lqr.f_max, self.lqr.f_max))
        return np.array([force], dtype=float)


@dataclass(frozen=True)
class FullCarLQRController:
    params: VehicleParams
    lqr: LQRParams
    name: str = "lqr_full"
    _gain: np.ndarray = field(init=False, repr=False)
    _closed_loop_poles: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        plant = FullCar(self.params)
        a_d, b_d = _zoh_discretize(plant.A, plant.B, self.lqr.T_s)
        # Outputs: [ddot z_s, phi, dot phi, theta, dot theta]
        c = np.zeros((5, 14), dtype=float)
        d = np.zeros((5, 4), dtype=float)
        c[0, :] = plant.A[1, :]
        d[0, :] = plant.B[1, :]
        c[1, 2] = 1.0  # phi
        c[2, 3] = 1.0  # dot phi
        c[3, 4] = 1.0  # theta
        c[4, 5] = 1.0  # dot theta
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
        gain, _ = _lqr_gain(a_d, b_d, c, d, q_y, r)
        poles = np.linalg.eigvals(a_d - b_d @ gain)
        object.__setattr__(self, "_gain", gain)
        object.__setattr__(self, "_closed_loop_poles", poles)

    @property
    def gain(self) -> np.ndarray:
        return self._gain

    @property
    def closed_loop_poles(self) -> np.ndarray:
        return self._closed_loop_poles

    def reset(self) -> None:
        return None

    def compute(self, state: ArrayLike, *_: Any, **__: Any) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (14,):
            raise ValueError(f"full-car state must have shape (14,), got {x.shape}.")
        force = -(self._gain @ x)
        if self.lqr.f_max is not None:
            force = np.clip(force, -self.lqr.f_max, self.lqr.f_max)
        return force.astype(float)
