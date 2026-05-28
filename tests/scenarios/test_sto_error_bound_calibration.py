from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.scenarios.sto_error_bound_calibration import (
    error_bound_calibration,
    run_phase_2_6,
)
from risk_aware_active_suspension.utils.config import ObserverParams


def test_error_bound_lookup_covers_measured_99_percentile(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    params = ObserverParams(lambda_1=100.0, lambda_2=2000.0, epsilon=0.02)

    result = error_bound_calibration(plant, params)

    assert result["default_bound_n"] > 0.0
    assert result["worst_case_bound_n"] >= result["default_bound_n"]
    assert result["max_relative_error"] <= 0.30
    for row in result["rows"]:
        assert row["coverage_99"] >= 0.989
        assert row["predicted_bound_n"] == row["bound_99_n"]


def test_phase_2_6_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_2_6(
        vehicle_config_path=repo_root / "configs" / "vehicle_default.yaml",
        observer_config_path=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "error_bound_lookup.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "predicted_vs_measured_bound.png").exists()
    assert (run_dir / "figures" / "error_bound_lookup.png").exists()
