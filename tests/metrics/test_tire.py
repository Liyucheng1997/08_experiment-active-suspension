import numpy as np
import pytest

from risk_aware_active_suspension.metrics.tire import (
    F_Z_MIN_DEFAULT,
    combined_horizontal_force,
    f_z_required,
    rho,
    rho_max,
)


def test_rho_hand_computed_scalar() -> None:
    # F_x = 300, F_y = 400 -> F_c = 500; mu*F_z = 0.9*5000 = 4500 -> rho = 1/9
    assert rho(300.0, 400.0, 5000.0, 0.9) == pytest.approx(500.0 / 4500.0)


def test_rho_at_saturation_limit() -> None:
    assert rho(1000.0, 0.0, 5000.0, 0.2) == pytest.approx(1.0)


def test_rho_zero_horizontal_demand_is_zero() -> None:
    assert rho(0.0, 0.0, 5000.0, 0.9) == 0.0


def test_rho_clamps_low_fz_to_avoid_nan() -> None:
    # F_z = 0 would be a divide by zero; expect floor F_z_min to kick in.
    value = rho(100.0, 0.0, 0.0, 0.9)
    assert np.isfinite(value)
    expected = 100.0 / (0.9 * F_Z_MIN_DEFAULT)
    assert value == pytest.approx(expected)


def test_rho_negative_fz_treated_as_floor() -> None:
    # Linear model can produce negative dynamic F_z; ρ must remain defined.
    value = rho(50.0, 0.0, -1000.0, 0.9)
    assert np.isfinite(value)
    assert value == pytest.approx(50.0 / (0.9 * F_Z_MIN_DEFAULT))


def test_rho_array_broadcasting() -> None:
    f_x = np.array([300.0, 0.0, 600.0])
    f_y = np.array([400.0, 500.0, 800.0])
    f_z = np.array([5000.0, 5000.0, 5000.0])
    mu = 0.9
    result = rho(f_x, f_y, f_z, mu)
    expected = np.sqrt(f_x**2 + f_y**2) / (mu * f_z)
    assert np.allclose(result, expected)


def test_rho_per_wheel_mu() -> None:
    # Per-wheel mu (e.g., split surface).
    f_x = np.array([1000.0, 1000.0])
    f_y = np.zeros(2)
    f_z = np.array([5000.0, 5000.0])
    mu = np.array([0.9, 0.4])
    result = rho(f_x, f_y, f_z, mu)
    assert result[0] == pytest.approx(1000.0 / (0.9 * 5000.0))
    assert result[1] == pytest.approx(1000.0 / (0.4 * 5000.0))


def test_rho_rejects_nonpositive_mu() -> None:
    with pytest.raises(ValueError):
        rho(100.0, 0.0, 5000.0, 0.0)
    with pytest.raises(ValueError):
        rho(100.0, 0.0, 5000.0, np.array([0.9, -0.1]))


def test_rho_rejects_nonpositive_fz_floor() -> None:
    with pytest.raises(ValueError):
        rho(100.0, 0.0, 5000.0, 0.9, f_z_min=0.0)


def test_rho_max_picks_worst_corner() -> None:
    f_x = np.array([100.0, 200.0, 50.0, 0.0])
    f_y = np.zeros(4)
    f_z = np.array([5000.0, 1000.0, 5000.0, 5000.0])
    mu = 0.9
    assert rho_max(f_x, f_y, f_z, mu) == pytest.approx(200.0 / (0.9 * 1000.0))


def test_f_z_required_zero_demand() -> None:
    assert f_z_required(0.0, 0.9, 0.85) == 0.0
    assert np.all(f_z_required(np.zeros(4), 0.9, 0.85) == 0.0)


def test_f_z_required_hand_computed() -> None:
    # F_c = 500, mu = 0.9, rho_safe = 0.5 -> F_z_required = 500 / (0.9 * 0.5) = 1111.11
    assert f_z_required(500.0, 0.9, 0.5) == pytest.approx(500.0 / 0.45)


def test_f_z_required_array_input() -> None:
    fc = np.array([0.0, 500.0, 1000.0])
    result = f_z_required(fc, 0.9, 0.85)
    assert np.allclose(result, fc / (0.9 * 0.85))


def test_f_z_required_rejects_nonpositive_rho_safe() -> None:
    with pytest.raises(ValueError):
        f_z_required(500.0, 0.9, 0.0)


def test_f_z_required_inverse_of_rho() -> None:
    # If F_z = F_z_required(F_c, mu, rho_safe), then ρ(F_x=F_c, F_y=0, F_z, mu) = rho_safe.
    f_c = 800.0
    mu = 0.95
    rho_safe = 0.85
    f_z_min = f_z_required(f_c, mu, rho_safe)
    assert rho(f_c, 0.0, f_z_min, mu) == pytest.approx(rho_safe)


def test_combined_horizontal_force() -> None:
    assert combined_horizontal_force(300.0, 400.0) == pytest.approx(500.0)
    assert np.allclose(
        combined_horizontal_force(np.array([3.0, 0.0]), np.array([4.0, 5.0])),
        np.array([5.0, 5.0]),
    )
