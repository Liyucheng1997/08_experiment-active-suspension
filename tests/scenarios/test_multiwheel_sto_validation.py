from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.multiwheel_sto_validation import (
    lateral_transfer_validation,
    run_phase_2_7,
    single_wheel_bump_validation,
)
from risk_aware_active_suspension.utils.config import ObserverParams


def test_single_wheel_bump_response_is_localized(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    params = ObserverParams(lambda_1=100.0, lambda_2=2000.0, epsilon=0.02)

    result = single_wheel_bump_validation(plant, params)

    assert result["estimate_peaks"][0] > 1000.0
    assert result["off_to_on_ratio"] < 0.05


def test_lateral_transfer_does_not_create_dynamic_sto_bias(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    params = ObserverParams(lambda_1=100.0, lambda_2=2000.0, epsilon=0.02)

    result = lateral_transfer_validation(plant, params)

    assert result["peak_lateral_transfer_n"] > 1000.0
    assert result["mean_to_transfer_ratio"] < 0.01


def test_phase_2_7_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_2_7(
        vehicle_config_path=repo_root / "configs" / "vehicle_default.yaml",
        observer_config_path=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "single_wheel_bump_estimates.png").exists()
    assert (run_dir / "figures" / "single_wheel_bump_truth.png").exists()
    assert (run_dir / "figures" / "lateral_transfer_dynamic_estimates.png").exists()
    assert (run_dir / "figures" / "lateral_transfer_fz_bar.png").exists()
