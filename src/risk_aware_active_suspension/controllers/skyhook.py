from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from risk_aware_active_suspension.utils.config import VehicleParams


@dataclass(frozen=True)
class QuarterCarSkyhookController:
    """Quarter-car skyhook: F_e = -c_sky * dot z_s, optionally saturated."""

    c_sky: float
    f_max: float | None = 4000.0
    name: str = "skyhook_quarter"

    def __post_init__(self) -> None:
        if self.c_sky < 0.0:
            raise ValueError("c_sky must be non-negative.")
        if self.f_max is not None and self.f_max <= 0.0:
            raise ValueError("f_max must be positive when provided.")

    def reset(self) -> None:
        return None

    def compute(self, state: ArrayLike, *_: Any, **__: Any) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (4,):
            raise ValueError(f"quarter-car state must have shape (4,), got {x.shape}.")
        force = -self.c_sky * x[1]
        if self.f_max is not None:
            force = float(np.clip(force, -self.f_max, self.f_max))
        return np.array([force], dtype=float)


@dataclass(frozen=True)
class FullCarSkyhookController:
    """Full-car per-corner skyhook on the body-corner vertical velocity.

    Corner velocity = dot z_s + y_ij * dot phi + x_ij * dot theta with the
    same FL/FR/RL/RR convention as FullCar.
    """

    params: VehicleParams
    c_sky: float
    f_max: float | None = 4000.0
    name: str = "skyhook_full"

    def __post_init__(self) -> None:
        if self.c_sky < 0.0:
            raise ValueError("c_sky must be non-negative.")
        if self.f_max is not None and self.f_max <= 0.0:
            raise ValueError("f_max must be positive when provided.")
        half_t = self.params.t / 2.0
        corners = np.array(
            [
                [self.params.l_f, half_t],
                [self.params.l_f, -half_t],
                [-self.params.l_r, half_t],
                [-self.params.l_r, -half_t],
            ],
            dtype=float,
        )
        object.__setattr__(self, "_corner_xy", corners)

    def reset(self) -> None:
        return None

    def compute(self, state: ArrayLike, *_: Any, **__: Any) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (14,):
            raise ValueError(f"full-car state must have shape (14,), got {x.shape}.")
        dz_s = x[1]
        dphi = x[3]
        dtheta = x[5]
        corner_vel = dz_s + self._corner_xy[:, 1] * dphi + self._corner_xy[:, 0] * dtheta
        force = -self.c_sky * corner_vel
        if self.f_max is not None:
            force = np.clip(force, -self.f_max, self.f_max)
        return force.astype(float)
