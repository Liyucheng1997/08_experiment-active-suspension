import numpy as np
import pytest

from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.load_transfer_validation import (
    load_transfer_validation,
    run_phase_1_5,
)


def test_quasi_static_longitudinal_transfer_matches_formula(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    result = load_transfer_validation(plant, a_x=5.0, a_y=5.0)

    assert result["front_delta"] == pytest.approx(result["expected_front_delta"], rel=0.05)
    assert result["rear_delta"] == pytest.approx(-result["expected_front_delta"], rel=0.05)
    assert result["front_delta_rel_error"] < 0.05


def test_quasi_static_lateral_transfer_matches_formula(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    result = load_transfer_validation(plant, a_x=5.0, a_y=5.0)

    assert result["left_delta"] == pytest.approx(result["expected_left_delta"], rel=0.05)
    assert result["right_delta"] == pytest.approx(-result["expected_left_delta"], rel=0.05)
    assert result["left_delta_rel_error"] < 0.05


def test_quasi_static_load_transfer_conserves_total_weight(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    static_total = np.sum(plant.static_loads())

    loads = plant.normal_loads_quasi_static(a_x=5.0, a_y=5.0)

    assert np.sum(loads) == pytest.approx(static_total, abs=1e-9)


def test_phase_1_5_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_1_5(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "longitudinal_load_transfer.png").exists()
    assert (run_dir / "figures" / "lateral_load_transfer.png").exists()
