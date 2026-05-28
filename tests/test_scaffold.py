from risk_aware_active_suspension import __version__


def test_package_imports() -> None:
    assert __version__


def test_default_vehicle_fixture(default_vehicle) -> None:
    assert default_vehicle.m == 1500.0
