from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from risk_aware_active_suspension.utils.config import VehicleParams


@dataclass(frozen=True)
class QuarterCar:
    params: VehicleParams

    def __post_init__(self) -> None:
        object.__setattr__(self, "m_s", self.params.sprung_mass_per_corner)
        object.__setattr__(self, "m_u", self.params.m_u)
        object.__setattr__(self, "A", self._build_A())
        object.__setattr__(self, "B", self._build_B())
        object.__setattr__(self, "E", self._build_E())

    def derivative(self, x: ArrayLike, u: float, w: float) -> np.ndarray:
        state = np.asarray(x, dtype=float)
        if state.shape != (4,):
            raise ValueError(f"x must have shape (4,), got {state.shape}.")
        return self.A @ state + self.B[:, 0] * float(u) + self.E[:, 0] * float(w)

    def step(self, x: ArrayLike, u: float, w: float, dt: float) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        state = np.asarray(x, dtype=float)
        k1 = self.derivative(state, u, w)
        k2 = self.derivative(state + 0.5 * dt * k1, u, w)
        k3 = self.derivative(state + 0.5 * dt * k2, u, w)
        k4 = self.derivative(state + dt * k3, u, w)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def simulate(
        self,
        t: ArrayLike,
        u_seq: ArrayLike,
        w_seq: ArrayLike,
        x0: ArrayLike | None = None,
    ) -> np.ndarray:
        time = np.asarray(t, dtype=float)
        if time.ndim != 1 or len(time) < 2:
            raise ValueError("t must be one-dimensional with at least two samples.")
        u = _as_sequence(u_seq, len(time), "u_seq")
        w = _as_sequence(w_seq, len(time), "w_seq")

        states = np.zeros((len(time), 4), dtype=float)
        if x0 is not None:
            states[0] = np.asarray(x0, dtype=float)
        for idx in range(len(time) - 1):
            dt = float(time[idx + 1] - time[idx])
            states[idx + 1] = self.step(states[idx], u[idx], w[idx], dt)
        return states

    def energy(self, x: ArrayLike, w: float = 0.0) -> float:
        z_s, v_s, z_u, v_u = np.asarray(x, dtype=float)
        dz_susp = z_s - z_u
        dz_tire = z_u - float(w)
        return float(
            0.5 * self.m_s * v_s**2
            + 0.5 * self.m_u * v_u**2
            + 0.5 * self.params.k_s * dz_susp**2
            + 0.5 * self.params.k_t * dz_tire**2
        )

    def sprung_acceleration(self, x: ArrayLike, u: float = 0.0, w: float = 0.0) -> float:
        return float(self.derivative(x, u=u, w=w)[1])

    def tire_normal_force(self, x: ArrayLike, w: float = 0.0, g: float = 9.81) -> float:
        state = np.asarray(x, dtype=float)
        dynamic_force = self.params.k_t * (state[2] - float(w))
        return float(self.params.m * g / 4.0 + dynamic_force)

    def _build_A(self) -> np.ndarray:
        p = self.params
        m_s = self.m_s
        m_u = self.m_u
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-p.k_s / m_s, -p.c_s / m_s, p.k_s / m_s, p.c_s / m_s],
                [0.0, 0.0, 0.0, 1.0],
                [p.k_s / m_u, p.c_s / m_u, -(p.k_s + p.k_t) / m_u, -p.c_s / m_u],
            ],
            dtype=float,
        )

    def _build_B(self) -> np.ndarray:
        return np.array([[0.0], [1.0 / self.m_s], [0.0], [-1.0 / self.m_u]], dtype=float)

    def _build_E(self) -> np.ndarray:
        return np.array([[0.0], [0.0], [0.0], [self.params.k_t / self.m_u]], dtype=float)


def _as_sequence(values: ArrayLike, expected_len: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(expected_len, float(arr), dtype=float)
    if arr.shape != (expected_len,):
        raise ValueError(f"{name} must be scalar or shape ({expected_len},), got {arr.shape}.")
    return arr
