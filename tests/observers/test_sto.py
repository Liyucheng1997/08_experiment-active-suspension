import pytest

from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.utils.config import ObserverParams


def test_sto_zero_error_keeps_zero_residual(default_vehicle) -> None:
    sto = STO(ObserverParams(lambda_1=20.0, lambda_2=50.0), unsprung_mass=default_vehicle.m_u)

    for _ in range(100):
        d_z_hat, f_z_hat = sto.step(v_u_meas=0.0, phi_known=0.0, dt=0.001, f_z_bar=123.0)

    assert sto.chi_hat == pytest.approx(0.0, abs=1e-12)
    assert d_z_hat == pytest.approx(0.0, abs=1e-12)
    assert f_z_hat == pytest.approx(123.0, abs=1e-12)


def test_sto_rejects_nonpositive_dt(default_vehicle) -> None:
    sto = STO(ObserverParams(), unsprung_mass=default_vehicle.m_u)

    with pytest.raises(ValueError, match="dt"):
        sto.step(v_u_meas=0.0, phi_known=0.0, dt=0.0)


def test_sto_saturation_switching(default_vehicle) -> None:
    sto = STO(ObserverParams(epsilon=0.1), unsprung_mass=default_vehicle.m_u, use_saturation=True)

    assert sto._switching(0.05) == pytest.approx(0.5)
    assert sto._switching(1.0) == pytest.approx(1.0)
    assert sto._switching(-1.0) == pytest.approx(-1.0)
