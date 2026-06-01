from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


F_Z_MIN_DEFAULT = 1.0  # Newtons; floor used to keep rho finite at light tire loads.


def rho(
    f_x: ArrayLike,
    f_y: ArrayLike,
    f_z: ArrayLike,
    mu: ArrayLike,
    f_z_min: float = F_Z_MIN_DEFAULT,
) -> np.ndarray:
    """Tire friction utilization ρ = sqrt(F_x² + F_y²) / (μ · max(F_z, F_z_min)).

    Inputs may be scalars or compatible numpy arrays. The F_z floor avoids
    division-by-zero (and NaN propagation) when a wheel goes near-airborne.
    A wheel hitting the floor is reported as ρ ≥ 1 / (μ · F_z_min · ...);
    callers should treat ρ ≥ 1 as saturation regardless.
    """
    if f_z_min <= 0.0:
        raise ValueError("f_z_min must be positive.")
    fx = np.asarray(f_x, dtype=float)
    fy = np.asarray(f_y, dtype=float)
    fz = np.asarray(f_z, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    if np.any(mu_arr <= 0.0):
        raise ValueError("mu must be strictly positive.")
    fz_eff = np.maximum(fz, f_z_min)
    return np.sqrt(fx * fx + fy * fy) / (mu_arr * fz_eff)


def rho_max(
    f_x_arr: ArrayLike,
    f_y_arr: ArrayLike,
    f_z_arr: ArrayLike,
    mu_arr: ArrayLike,
    f_z_min: float = F_Z_MIN_DEFAULT,
) -> float:
    """Peak utilization across a set of tires (e.g., 4 corners).

    Inputs broadcast together. Returns the scalar maximum of ρ_ij.
    """
    rho_all = rho(f_x_arr, f_y_arr, f_z_arr, mu_arr, f_z_min=f_z_min)
    return float(np.max(rho_all))


def f_z_required(
    f_c: ArrayLike,
    mu: ArrayLike,
    rho_safe: float,
) -> np.ndarray:
    """Minimum normal load that keeps utilization ≤ ρ_safe given combined horizontal demand.

    F_z_required = F_c / (μ · ρ_safe). With F_c = 0, returns 0 (no support needed).
    """
    if rho_safe <= 0.0:
        raise ValueError("rho_safe must be strictly positive.")
    fc = np.asarray(f_c, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    if np.any(mu_arr <= 0.0):
        raise ValueError("mu must be strictly positive.")
    return fc / (mu_arr * rho_safe)


def combined_horizontal_force(f_x: ArrayLike, f_y: ArrayLike) -> np.ndarray:
    fx = np.asarray(f_x, dtype=float)
    fy = np.asarray(f_y, dtype=float)
    return np.sqrt(fx * fx + fy * fy)
