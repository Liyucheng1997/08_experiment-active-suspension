from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.scenarios.sto_real_road_validation import (
    run_phase_2_4,
    sto_real_road_validation,
)
from risk_aware_active_suspension.utils.config import ObserverParams


def test_sto_tracks_quarter_car_real_road_residual(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    params = ObserverParams(lambda_1=100.0, lambda_2=2000.0)

    result = sto_real_road_validation(plant, params, duration=6.0)

    assert result["correlation_after_transient"] > 0.9
    assert result["rmse_ratio_after_transient"] < 0.15


def test_phase_2_4_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_2_4(
        vehicle_config_path=repo_root / "configs" / "vehicle_default.yaml",
        observer_config_path=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "tire_dynamic_force_tracking.png").exists()
    assert (run_dir / "figures" / "tire_dynamic_force_error.png").exists()
    assert (run_dir / "figures" / "road_and_velocity.png").exists()
