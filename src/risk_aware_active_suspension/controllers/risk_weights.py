from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


# Hard clip on the sigmoid argument: |k_rho * (rho - rho_th)| <= SIGMOID_ARG_LIMIT.
# Beyond this the sigmoid is numerically saturated to {0, 1} anyway; clipping
# guards exp() against overflow on adversarial inputs (e.g. NaN-ish ρ).
SIGMOID_ARG_LIMIT = 50.0


@dataclass(frozen=True)
class RiskWeightParams:
    """Bundled parameters for the sigma_rho scheduler + global/local weights."""

    rho_th: float = 0.75
    k_rho: float = 25.0
    kappa_rho: float = 5.0
    q_c_min: float = 1.0
    q_c_max: float = 1.0
    q_p_min: float = 0.0
    q_p_max: float = 1.0

    def __post_init__(self) -> None:
        if self.rho_th <= 0.0:
            raise ValueError("rho_th must be positive.")
        if self.k_rho <= 0.0:
            raise ValueError("k_rho must be positive.")
        if self.kappa_rho < 0.0:
            raise ValueError("kappa_rho must be non-negative.")
        if self.q_c_min < 0.0 or self.q_c_max < self.q_c_min:
            raise ValueError("q_c bounds must satisfy 0 <= q_c_min <= q_c_max.")
        if self.q_p_min < 0.0 or self.q_p_max < self.q_p_min:
            raise ValueError("q_p bounds must satisfy 0 <= q_p_min <= q_p_max.")


def sigma_rho(rho_max: ArrayLike, rho_th: float, k_rho: float) -> np.ndarray | float:
    """Sigmoid risk activation σ(ρ_max) ∈ [0, 1] centered at ρ_th, slope k_rho.

    σ(ρ_th) = 0.5 exactly. Monotone increasing in ρ_max. The argument is clipped
    to ±SIGMOID_ARG_LIMIT before exp() to prevent overflow on extreme inputs.
    """
    if rho_th <= 0.0:
        raise ValueError("rho_th must be positive.")
    if k_rho <= 0.0:
        raise ValueError("k_rho must be positive.")
    arr = np.asarray(rho_max, dtype=float)
    z = np.clip(k_rho * (arr - rho_th), -SIGMOID_ARG_LIMIT, SIGMOID_ARG_LIMIT)
    out = 1.0 / (1.0 + np.exp(-z))
    return float(out) if arr.ndim == 0 else out


def q_c(sigma: ArrayLike, q_c_min: float, q_c_max: float) -> np.ndarray | float:
    """Global comfort weight: linear interpolation, comfort fades as risk rises.

    σ=0 -> q_c = q_c_max (full comfort); σ=1 -> q_c = q_c_min (comfort suppressed).
    """
    if q_c_max < q_c_min:
        raise ValueError("Require q_c_min <= q_c_max.")
    s = np.clip(np.asarray(sigma, dtype=float), 0.0, 1.0)
    out = q_c_max + s * (q_c_min - q_c_max)
    return float(out) if s.ndim == 0 else out


def q_p(sigma: ArrayLike, q_p_min: float, q_p_max: float) -> np.ndarray | float:
    """Global protection weight: rises linearly with σ.

    σ=0 -> q_p = q_p_min (off / comfort-only); σ=1 -> q_p = q_p_max.
    """
    if q_p_max < q_p_min:
        raise ValueError("Require q_p_min <= q_p_max.")
    s = np.clip(np.asarray(sigma, dtype=float), 0.0, 1.0)
    out = q_p_min + s * (q_p_max - q_p_min)
    return float(out) if s.ndim == 0 else out


def q_p_local(
    sigma: ArrayLike,
    rho_ij: ArrayLike,
    rho_th: float,
    kappa_rho: float,
    q_p_min: float = 0.0,
    q_p_max: float = 1.0,
) -> np.ndarray:
    """Per-wheel protection weight: global q_p(σ) amplified on wheels with ρ_ij > ρ_th.

    q_{p,ij} = q_p(σ) · (1 + κ_ρ · max(ρ_ij - ρ_th, 0))

    With κ_ρ = 0 reduces to a uniform global q_p (ablation A3 in Phase 6.6).
    With q_p_max = 0 reduces identically to zero (acceptance: cost is comfort-only).
    """
    if kappa_rho < 0.0:
        raise ValueError("kappa_rho must be non-negative.")
    base = q_p(sigma, q_p_min, q_p_max)
    rho = np.asarray(rho_ij, dtype=float)
    amplification = 1.0 + kappa_rho * np.maximum(rho - rho_th, 0.0)
    return base * amplification


def compute_all(
    rho_max_value: float,
    rho_ij: ArrayLike,
    params: RiskWeightParams,
) -> dict[str, float | np.ndarray]:
    """One-call helper: takes a single ρ_max + per-wheel ρ_ij, returns the bundle.

    Useful inside controller compute loops where all four quantities are needed.
    """
    s = sigma_rho(rho_max_value, params.rho_th, params.k_rho)
    return {
        "sigma": s,
        "q_c": q_c(s, params.q_c_min, params.q_c_max),
        "q_p": q_p(s, params.q_p_min, params.q_p_max),
        "q_p_local": q_p_local(
            s, rho_ij, params.rho_th, params.kappa_rho, params.q_p_min, params.q_p_max
        ),
    }
