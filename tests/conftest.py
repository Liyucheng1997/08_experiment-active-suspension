from pathlib import Path

import pytest

from risk_aware_active_suspension.utils.config import VehicleParams, from_yaml


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def default_vehicle(repo_root: Path) -> VehicleParams:
    return from_yaml(repo_root / "configs" / "vehicle_default.yaml")
