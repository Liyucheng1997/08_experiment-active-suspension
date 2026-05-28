from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from risk_aware_active_suspension.utils.config import VehicleParams


@dataclass(frozen=True)
class HalfCar:
    """Heave/roll half-car with left/right aggregate unsprung masses.

    Each side represents the front and rear corner pair. The suspension and
    unsprung parameters are therefore doubled per side while the sprung body
    uses the full vehicle mass and roll inertia.
    """

    params: VehicleParams

    def __post_init__(self) -> None:
        object.__setattr__(self, "m_s", self.params.m)
        object.__setattr__(self, "m_u_side", 2.0 * self.params.m_u)
        object.__setattr__(self, "k_side", 2.0 * self.params.k_s)
        object.__setattr__(self, "c_side", 2.0 * self.params.c_s)
        object.__setattr__(self, "y_left", self.params.t / 2.0)
        object.__setattr__(self, "y_right", -self.params.t / 2.0)
        object.__setattr__(self, "A", self._build_A())
        object.__setattr__(self, "E", self._build_E())

    def derivative(self, x: ArrayLike, w: ArrayLike) -> np.ndarray:
        state = np.asarray(x, dtype=float)
        road = np.asarray(w, dtype=float)
        if state.shape != (8,):
            raise ValueError(f"x must have shape (8,), got {state.shape}.")
        if road.shape != (2,):
            raise ValueError(f"w must have shape (2,), got {road.shape}.")
        return self.A @ state + self.E @ road

    def step(self, x: ArrayLike, w: ArrayLike, dt: float) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        state = np.asarray(x, dtype=float)
        road = np.asarray(w, dtype=float)
        k1 = self.derivative(state, road)
        k2 = self.derivative(state + 0.5 * dt * k1, road)
        k3 = self.derivative(state + 0.5 * dt * k2, road)
        k4 = self.derivative(state + dt * k3, road)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def simulate(self, t: ArrayLike, w_seq: ArrayLike, x0: ArrayLike | None = None) -> np.ndarray:
        time = np.asarray(t, dtype=float)
        if time.ndim != 1 or len(time) < 2:
            raise ValueError("t must be one-dimensional with at least two samples.")
        road = np.asarray(w_seq, dtype=float)
        if road.shape != (len(time), 2):
            raise ValueError(f"w_seq must have shape ({len(time)}, 2), got {road.shape}.")

        states = np.zeros((len(time), 8), dtype=float)
        if x0 is not None:
            states[0] = np.asarray(x0, dtype=float)
        for idx in range(len(time) - 1):
            dt = float(time[idx + 1] - time[idx])
            states[idx + 1] = self.step(states[idx], road[idx], dt)
        return states

    def analytical_roll_natural_frequency(self) -> float:
        k_roll = self.k_side * self.y_left**2 + self.k_side * self.y_right**2
        return float(np.sqrt(k_roll / self.params.I_x))

    def _build_A(self) -> np.ndarray:
        a = np.zeros((8, 8), dtype=float)
        a[0, 1] = 1.0
        a[2, 3] = 1.0
        a[4, 5] = 1.0
        a[6, 7] = 1.0

        self._add_side_to_A(a, z_u_idx=4, v_u_idx=5, y=self.y_left)
        self._add_side_to_A(a, z_u_idx=6, v_u_idx=7, y=self.y_right)
        return a

    def _add_side_to_A(self, a: np.ndarray, z_u_idx: int, v_u_idx: int, y: float) -> None:
        k = self.k_side
        c = self.c_side
        m_s = self.m_s
        i_x = self.params.I_x
        m_u = self.m_u_side
        k_t = 2.0 * self.params.k_t

        # Suspension deflection: d = z_s + y*phi - z_u.
        a[1, 0] += -k / m_s
        a[1, 1] += -c / m_s
        a[1, 2] += -k * y / m_s
        a[1, 3] += -c * y / m_s
        a[1, z_u_idx] += k / m_s
        a[1, v_u_idx] += c / m_s

        a[3, 0] += -k * y / i_x
        a[3, 1] += -c * y / i_x
        a[3, 2] += -k * y * y / i_x
        a[3, 3] += -c * y * y / i_x
        a[3, z_u_idx] += k * y / i_x
        a[3, v_u_idx] += c * y / i_x

        a[v_u_idx, 0] += k / m_u
        a[v_u_idx, 1] += c / m_u
        a[v_u_idx, 2] += k * y / m_u
        a[v_u_idx, 3] += c * y / m_u
        a[v_u_idx, z_u_idx] += -(k + k_t) / m_u
        a[v_u_idx, v_u_idx] += -c / m_u

    def _build_E(self) -> np.ndarray:
        e = np.zeros((8, 2), dtype=float)
        k_t = 2.0 * self.params.k_t
        e[5, 0] = k_t / self.m_u_side
        e[7, 1] = k_t / self.m_u_side
        return e
