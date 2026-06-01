import numpy as np
import pytest

from risk_aware_active_suspension.controllers.risk_weights import (
    RiskWeightParams,
    compute_all,
    q_c,
    q_p,
    q_p_local,
    sigma_rho,
)


# ---- sigma_rho ----------------------------------------------------------


def test_sigma_at_threshold_is_one_half() -> None:
    assert sigma_rho(0.75, rho_th=0.75, k_rho=25.0) == pytest.approx(0.5)


def test_sigma_monotone_increasing_in_rho() -> None:
    rho_grid = np.linspace(0.0, 1.5, 50)
    s = sigma_rho(rho_grid, rho_th=0.75, k_rho=25.0)
    assert np.all(np.diff(s) >= 0.0)


def test_sigma_limits_to_zero_and_one() -> None:
    assert sigma_rho(0.0, 0.75, 25.0) < 1e-6
    assert sigma_rho(2.0, 0.75, 25.0) > 1.0 - 1e-6


def test_sigma_bounded_in_unit_interval() -> None:
    s = sigma_rho(np.linspace(-5.0, 5.0, 200), rho_th=0.75, k_rho=25.0)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_sigma_input_clipping_prevents_overflow() -> None:
    # Without clipping, k_rho * (rho - rho_th) = 25 * 1e9 -> exp(...) overflows.
    s = sigma_rho(1.0e9, rho_th=0.75, k_rho=25.0)
    assert np.isfinite(s) and s == pytest.approx(1.0)
    s = sigma_rho(-1.0e9, rho_th=0.75, k_rho=25.0)
    assert np.isfinite(s) and s == pytest.approx(0.0)


def test_sigma_smoothness_no_jumps() -> None:
    rho_grid = np.linspace(0.0, 1.5, 2000)
    s = sigma_rho(rho_grid, 0.75, 25.0)
    # Max derivative ≈ k_rho / 4 at the center (= 6.25). Numerical diff stays bounded.
    d = np.diff(s) / np.diff(rho_grid)
    assert np.max(d) < 25.0 / 4.0 + 0.1


def test_sigma_scalar_returns_scalar() -> None:
    assert isinstance(sigma_rho(0.5, 0.75, 25.0), float)


def test_sigma_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        sigma_rho(0.5, rho_th=0.0, k_rho=25.0)
    with pytest.raises(ValueError):
        sigma_rho(0.5, rho_th=0.75, k_rho=0.0)


# ---- q_c / q_p ----------------------------------------------------------


def test_q_c_at_endpoints() -> None:
    assert q_c(0.0, q_c_min=0.1, q_c_max=1.0) == pytest.approx(1.0)
    assert q_c(1.0, q_c_min=0.1, q_c_max=1.0) == pytest.approx(0.1)


def test_q_c_monotone_decreasing() -> None:
    s = np.linspace(0.0, 1.0, 50)
    values = q_c(s, q_c_min=0.1, q_c_max=1.0)
    assert np.all(np.diff(values) <= 0.0)


def test_q_p_at_endpoints() -> None:
    assert q_p(0.0, q_p_min=0.0, q_p_max=5.0) == 0.0
    assert q_p(1.0, q_p_min=0.0, q_p_max=5.0) == pytest.approx(5.0)


def test_q_p_monotone_increasing() -> None:
    s = np.linspace(0.0, 1.0, 50)
    values = q_p(s, q_p_min=0.0, q_p_max=5.0)
    assert np.all(np.diff(values) >= 0.0)


def test_q_c_and_q_p_clip_sigma_outside_unit() -> None:
    # σ > 1 should saturate at q_c_min / q_p_max, not extrapolate.
    assert q_c(5.0, 0.1, 1.0) == pytest.approx(0.1)
    assert q_p(-1.0, 0.0, 5.0) == 0.0


def test_q_p_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        q_p(0.5, q_p_min=2.0, q_p_max=1.0)


# ---- q_p_local ----------------------------------------------------------


def test_q_p_local_no_amplification_below_threshold() -> None:
    # ρ_ij ≤ ρ_th -> amp = 1 -> local equals global q_p(σ).
    sigma = 0.6
    base = q_p(sigma, q_p_min=0.0, q_p_max=10.0)
    out = q_p_local(sigma, np.array([0.4, 0.5, 0.7, 0.75]),
                    rho_th=0.75, kappa_rho=5.0, q_p_min=0.0, q_p_max=10.0)
    assert np.allclose(out, base)


def test_q_p_local_amplifies_per_wheel_overshoot() -> None:
    sigma = 1.0
    out = q_p_local(sigma, np.array([0.75, 0.85, 0.95, 1.05]),
                    rho_th=0.75, kappa_rho=5.0, q_p_min=0.0, q_p_max=2.0)
    # base q_p(1) = 2; amp = 1 + 5*[0, 0.10, 0.20, 0.30]
    expected = 2.0 * (1.0 + 5.0 * np.array([0.0, 0.10, 0.20, 0.30]))
    assert np.allclose(out, expected)


def test_q_p_local_kappa_zero_reduces_to_uniform_global() -> None:
    # κ_ρ = 0 disables local amplification (ablation A3 in Phase 6.6).
    out = q_p_local(0.7, np.array([0.5, 0.9, 1.2, 0.6]),
                    rho_th=0.75, kappa_rho=0.0, q_p_min=0.0, q_p_max=4.0)
    assert np.allclose(out, q_p(0.7, 0.0, 4.0))


def test_q_p_max_zero_kills_protection() -> None:
    # Acceptance: with q_p ≡ 0 the cost reduces to comfort-only.
    out = q_p_local(1.0, np.array([0.5, 0.8, 1.1, 1.5]),
                    rho_th=0.75, kappa_rho=5.0, q_p_min=0.0, q_p_max=0.0)
    assert np.all(out == 0.0)


def test_q_p_local_rejects_negative_kappa() -> None:
    with pytest.raises(ValueError):
        q_p_local(0.5, 0.8, 0.75, kappa_rho=-1.0)


# ---- RiskWeightParams + compute_all -------------------------------------


def test_risk_weight_params_validation() -> None:
    RiskWeightParams()  # default ctor
    with pytest.raises(ValueError):
        RiskWeightParams(rho_th=0.0)
    with pytest.raises(ValueError):
        RiskWeightParams(k_rho=-1.0)
    with pytest.raises(ValueError):
        RiskWeightParams(kappa_rho=-0.1)
    with pytest.raises(ValueError):
        RiskWeightParams(q_p_min=2.0, q_p_max=1.0)


def test_compute_all_consistency() -> None:
    p = RiskWeightParams(rho_th=0.75, k_rho=25.0, kappa_rho=5.0,
                         q_c_min=0.1, q_c_max=1.0, q_p_min=0.0, q_p_max=2.0)
    rho_ij = np.array([0.4, 0.85, 0.95, 0.5])
    rho_max_value = float(np.max(rho_ij))
    bundle = compute_all(rho_max_value, rho_ij, p)
    assert bundle["sigma"] == pytest.approx(sigma_rho(rho_max_value, p.rho_th, p.k_rho))
    assert bundle["q_c"] == pytest.approx(q_c(bundle["sigma"], p.q_c_min, p.q_c_max))
    assert bundle["q_p"] == pytest.approx(q_p(bundle["sigma"], p.q_p_min, p.q_p_max))
    assert np.allclose(
        bundle["q_p_local"],
        q_p_local(bundle["sigma"], rho_ij, p.rho_th, p.kappa_rho, p.q_p_min, p.q_p_max),
    )


def test_compute_all_comfort_only_when_q_p_max_zero() -> None:
    p = RiskWeightParams(q_p_min=0.0, q_p_max=0.0)
    rho_ij = np.array([0.4, 0.9, 1.2, 0.5])
    bundle = compute_all(float(np.max(rho_ij)), rho_ij, p)
    assert bundle["q_p"] == 0.0
    assert np.all(bundle["q_p_local"] == 0.0)
