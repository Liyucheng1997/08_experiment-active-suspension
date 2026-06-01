from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from risk_aware_active_suspension.utils.config import VehicleParams


CORNER_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class FullCar:
    """Linear 7-DOF full-car vertical dynamics model.

    State order:
    [z_s, dz_s, phi, dphi, theta, dtheta, z_u_FL, dz_u_FL, ..., z_u_RR, dz_u_RR].

    Corner order is FL, FR, RL, RR. The body corner displacement convention is
    z_corner = z_s + y*phi + x*theta, where x is positive forward and y is
    positive left.
    """

    params: VehicleParams

    def __post_init__(self) -> None:
        object.__setattr__(self, "m_s", self.params.m)
        object.__setattr__(self, "corner_xy", self._corner_coordinates())
        object.__setattr__(self, "corner_k", self._corner_stiffnesses())
        object.__setattr__(self, "corner_c", self._corner_dampings())
        object.__setattr__(self, "A", self._build_A())
        object.__setattr__(self, "B", self._build_B())
        object.__setattr__(self, "E", self._build_E())

    def static_loads(self, g: float = 9.81) -> np.ndarray:
        p = self.params
        wheelbase = p.l_f + p.l_r
        front_total = p.m * g * p.l_r / wheelbase
        rear_total = p.m * g * p.l_f / wheelbase
        return np.array(
            [
                front_total / 2.0,
                front_total / 2.0,
                rear_total / 2.0,
                rear_total / 2.0,
            ],
            dtype=float,
        )

    def normal_loads_quasi_static(self, a_x: float = 0.0, a_y: float = 0.0, g: float = 9.81) -> np.ndarray:
        """Normal loads with quasi-static longitudinal and lateral transfer.

        Positive a_x is forward acceleration, so front axle normal load
        decreases. Positive a_y transfers load to the left side under the
        coordinate convention used by the roll model.
        """
        p = self.params
        loads = self.static_loads(g=g)
        wheelbase = p.l_f + p.l_r
        longitudinal_front_delta = -p.m * p.h_g * float(a_x) / wheelbase
        lateral_left_delta = p.m * p.h_g * float(a_y) / p.t

        loads[0:2] += longitudinal_front_delta / 2.0
        loads[2:4] -= longitudinal_front_delta / 2.0
        loads[[0, 2]] += lateral_left_delta / 2.0
        loads[[1, 3]] -= lateral_left_delta / 2.0
        return loads

    def derivative(
        self,
        x: ArrayLike,
        w: ArrayLike,
        u: ArrayLike | None = None,
        body_acc: ArrayLike | None = None,
    ) -> np.ndarray:
        state = np.asarray(x, dtype=float)
        road = np.asarray(w, dtype=float)
        if state.shape != (14,):
            raise ValueError(f"x must have shape (14,), got {state.shape}.")
        if road.shape != (4,):
            raise ValueError(f"w must have shape (4,), got {road.shape}.")
        out = self.A @ state + self.E @ road
        if u is not None:
            force = np.asarray(u, dtype=float)
            if force.shape != (4,):
                raise ValueError(f"u must have shape (4,), got {force.shape}.")
            out = out + self.B @ force
        if body_acc is not None:
            acc = np.asarray(body_acc, dtype=float)
            if acc.shape != (2,):
                raise ValueError(f"body_acc must have shape (2,) = (a_x, a_y), got {acc.shape}.")
            # Body-frame inertial reaction: forward accel a_x lifts the nose
            # (pitch moment M_y = m*h_g*a_x); lateral accel a_y rolls the body
            # away from the turn (M_x = -m*h_g*a_y) so that load transfers
            # consistently with normal_loads_quasi_static.
            p = self.params
            out[3] += -p.m * p.h_g * acc[1] / p.I_x
            out[5] += p.m * p.h_g * acc[0] / p.I_y
        return out

    def step(
        self,
        x: ArrayLike,
        w: ArrayLike,
        dt: float,
        u: ArrayLike | None = None,
        body_acc: ArrayLike | None = None,
    ) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        state = np.asarray(x, dtype=float)
        road = np.asarray(w, dtype=float)
        k1 = self.derivative(state, road, u, body_acc)
        k2 = self.derivative(state + 0.5 * dt * k1, road, u, body_acc)
        k3 = self.derivative(state + 0.5 * dt * k2, road, u, body_acc)
        k4 = self.derivative(state + dt * k3, road, u, body_acc)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def simulate(
        self,
        t: ArrayLike,
        w_seq: ArrayLike,
        x0: ArrayLike | None = None,
        u_seq: ArrayLike | None = None,
    ) -> np.ndarray:
        time = np.asarray(t, dtype=float)
        if time.ndim != 1 or len(time) < 2:
            raise ValueError("t must be one-dimensional with at least two samples.")
        road = np.asarray(w_seq, dtype=float)
        if road.shape != (len(time), 4):
            raise ValueError(f"w_seq must have shape ({len(time)}, 4), got {road.shape}.")
        if u_seq is None:
            forces: np.ndarray | None = None
        else:
            forces = np.asarray(u_seq, dtype=float)
            if forces.shape != (len(time), 4):
                raise ValueError(f"u_seq must have shape ({len(time)}, 4), got {forces.shape}.")

        states = np.zeros((len(time), 14), dtype=float)
        if x0 is not None:
            states[0] = np.asarray(x0, dtype=float)
        for idx in range(len(time) - 1):
            dt = float(time[idx + 1] - time[idx])
            u_now = None if forces is None else forces[idx]
            states[idx + 1] = self.step(states[idx], road[idx], dt, u=u_now)
        return states

    def tire_normal_forces(self, x: ArrayLike, w: ArrayLike, g: float = 9.81) -> np.ndarray:
        state = np.asarray(x, dtype=float)
        road = np.asarray(w, dtype=float)
        forces = self.static_loads(g=g).copy()
        for corner_idx in range(4):
            z_u_idx = 6 + 2 * corner_idx
            forces[corner_idx] += self.params.k_t * (state[z_u_idx] - road[corner_idx])
        return forces

    def suspension_strokes(self, x: ArrayLike) -> np.ndarray:
        state = np.asarray(x, dtype=float)
        strokes = np.zeros(4, dtype=float)
        for corner_idx, (x_pos, y_pos) in enumerate(self.corner_xy):
            z_u_idx = 6 + 2 * corner_idx
            body_corner_z = state[0] + y_pos * state[2] + x_pos * state[4]
            strokes[corner_idx] = body_corner_z - state[z_u_idx]
        return strokes

    def body_accelerations(
        self,
        x: ArrayLike,
        w: ArrayLike,
        u: ArrayLike | None = None,
        body_acc: ArrayLike | None = None,
    ) -> np.ndarray:
        dx = self.derivative(x, w, u=u, body_acc=body_acc)
        return np.array([dx[1], dx[3], dx[5]], dtype=float)

    def batch_tire_normal_forces(self, states: ArrayLike, roads: ArrayLike, g: float = 9.81) -> np.ndarray:
        state_arr = np.asarray(states, dtype=float)
        road_arr = np.asarray(roads, dtype=float)
        return np.array([self.tire_normal_forces(x, w, g=g) for x, w in zip(state_arr, road_arr)])

    def batch_suspension_strokes(self, states: ArrayLike) -> np.ndarray:
        return np.array([self.suspension_strokes(x) for x in np.asarray(states, dtype=float)])

    def batch_body_accelerations(
        self, states: ArrayLike, roads: ArrayLike, forces: ArrayLike | None = None
    ) -> np.ndarray:
        state_arr = np.asarray(states, dtype=float)
        road_arr = np.asarray(roads, dtype=float)
        if forces is None:
            return np.array(
                [self.body_accelerations(x, w) for x, w in zip(state_arr, road_arr)]
            )
        force_arr = np.asarray(forces, dtype=float)
        return np.array(
            [
                self.body_accelerations(x, w, u=u)
                for x, w, u in zip(state_arr, road_arr, force_arr)
            ]
        )

    def analytical_roll_natural_frequency(self) -> float:
        k_roll = sum(k * y**2 for k, (_, y) in zip(self.corner_k, self.corner_xy))
        return float(np.sqrt(k_roll / self.params.I_x))

    def analytical_pitch_natural_frequency(self) -> float:
        k_pitch = sum(k * x**2 for k, (x, _) in zip(self.corner_k, self.corner_xy))
        return float(np.sqrt(k_pitch / self.params.I_y))

    def _corner_coordinates(self) -> tuple[tuple[float, float], ...]:
        p = self.params
        half_track = p.t / 2.0
        return (
            (p.l_f, half_track),
            (p.l_f, -half_track),
            (-p.l_r, half_track),
            (-p.l_r, -half_track),
        )

    def _corner_stiffnesses(self) -> tuple[float, float, float, float]:
        p = self.params
        wheelbase = p.l_f + p.l_r
        k_front = 2.0 * p.k_s * p.l_r / wheelbase
        k_rear = 2.0 * p.k_s * p.l_f / wheelbase
        return (k_front, k_front, k_rear, k_rear)

    def _corner_dampings(self) -> tuple[float, float, float, float]:
        p = self.params
        wheelbase = p.l_f + p.l_r
        c_front = 2.0 * p.c_s * p.l_r / wheelbase
        c_rear = 2.0 * p.c_s * p.l_f / wheelbase
        return (c_front, c_front, c_rear, c_rear)

    def _build_A(self) -> np.ndarray:
        a = np.zeros((14, 14), dtype=float)
        a[0, 1] = 1.0
        a[2, 3] = 1.0
        a[4, 5] = 1.0
        for corner_idx in range(4):
            a[6 + 2 * corner_idx, 7 + 2 * corner_idx] = 1.0
            self._add_corner_to_A(a, corner_idx)
        return a

    def _add_corner_to_A(self, a: np.ndarray, corner_idx: int) -> None:
        k = self.corner_k[corner_idx]
        c = self.corner_c[corner_idx]
        m_s = self.m_s
        i_x = self.params.I_x
        i_y = self.params.I_y
        m_u = self.params.m_u
        k_t = self.params.k_t
        x, y = self.corner_xy[corner_idx]
        z_u_idx = 6 + 2 * corner_idx
        v_u_idx = z_u_idx + 1

        # Suspension deflection: d = z_s + y*phi + x*theta - z_u.
        a[1, 0] += -k / m_s
        a[1, 1] += -c / m_s
        a[1, 2] += -k * y / m_s
        a[1, 3] += -c * y / m_s
        a[1, 4] += -k * x / m_s
        a[1, 5] += -c * x / m_s
        a[1, z_u_idx] += k / m_s
        a[1, v_u_idx] += c / m_s

        a[3, 0] += -k * y / i_x
        a[3, 1] += -c * y / i_x
        a[3, 2] += -k * y * y / i_x
        a[3, 3] += -c * y * y / i_x
        a[3, 4] += -k * x * y / i_x
        a[3, 5] += -c * x * y / i_x
        a[3, z_u_idx] += k * y / i_x
        a[3, v_u_idx] += c * y / i_x

        a[5, 0] += -k * x / i_y
        a[5, 1] += -c * x / i_y
        a[5, 2] += -k * x * y / i_y
        a[5, 3] += -c * x * y / i_y
        a[5, 4] += -k * x * x / i_y
        a[5, 5] += -c * x * x / i_y
        a[5, z_u_idx] += k * x / i_y
        a[5, v_u_idx] += c * x / i_y

        a[v_u_idx, 0] += k / m_u
        a[v_u_idx, 1] += c / m_u
        a[v_u_idx, 2] += k * y / m_u
        a[v_u_idx, 3] += c * y / m_u
        a[v_u_idx, 4] += k * x / m_u
        a[v_u_idx, 5] += c * x / m_u
        a[v_u_idx, z_u_idx] += -(k + k_t) / m_u
        a[v_u_idx, v_u_idx] += -c / m_u

    def _build_B(self) -> np.ndarray:
        """Per-corner actuator force input matrix.

        F_e_ij acts as +F_e on the sprung body at corner ij and -F_e on the
        unsprung mass of that corner, by Newton's third law on the suspension.
        Body equations distribute the corner force via (1/m_s, y/I_x, x/I_y).
        """
        b = np.zeros((14, 4), dtype=float)
        m_s = self.m_s
        i_x = self.params.I_x
        i_y = self.params.I_y
        m_u = self.params.m_u
        for corner_idx, (x_pos, y_pos) in enumerate(self.corner_xy):
            b[1, corner_idx] += 1.0 / m_s
            b[3, corner_idx] += y_pos / i_x
            b[5, corner_idx] += x_pos / i_y
            b[7 + 2 * corner_idx, corner_idx] += -1.0 / m_u
        return b

    def _build_E(self) -> np.ndarray:
        e = np.zeros((14, 4), dtype=float)
        for corner_idx in range(4):
            e[7 + 2 * corner_idx, corner_idx] = self.params.k_t / self.params.m_u
        return e
