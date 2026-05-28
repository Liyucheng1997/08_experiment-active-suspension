from pathlib import Path

import pytest

from risk_aware_active_suspension.utils.config import VehicleParams, from_yaml


def test_loads_default_vehicle_as_typed_params(repo_root: Path) -> None:
    params = from_yaml(repo_root / "configs" / "vehicle_default.yaml")

    assert isinstance(params, VehicleParams)
    assert all(isinstance(value, float) for value in params.__dict__.values())
    assert params.m == pytest.approx(1500.0)
    assert params.wheelbase == pytest.approx(2.6)
    assert params.sprung_mass_per_corner == pytest.approx(375.0)


def test_default_vehicle_bounds(default_vehicle: VehicleParams) -> None:
    assert 800.0 <= default_vehicle.m <= 2500.0
    assert 0.8 <= default_vehicle.l_f <= 2.0
    assert 0.8 <= default_vehicle.l_r <= 2.0
    assert 0.3 <= default_vehicle.h_g <= 0.9
    assert 1.2 <= default_vehicle.t <= 2.0
    assert 10_000.0 <= default_vehicle.k_s <= 80_000.0
    assert 500.0 <= default_vehicle.c_s <= 8_000.0
    assert 20.0 <= default_vehicle.m_u <= 100.0
    assert 100_000.0 <= default_vehicle.k_t <= 400_000.0
    assert 100.0 <= default_vehicle.I_x <= 2_000.0
    assert 500.0 <= default_vehicle.I_y <= 5_000.0


def test_rejects_unknown_vehicle_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad_vehicle.yaml"
    path.write_text(
        """
vehicle:
  m: 1500
  l_f: 1.2
  l_r: 1.4
  h_g: 0.55
  t: 1.55
  k_s: 30000
  c_s: 2000
  m_u: 50
  k_t: 200000
  I_x: 600
  I_y: 2400
  typo_field: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra"):
        from_yaml(path)
