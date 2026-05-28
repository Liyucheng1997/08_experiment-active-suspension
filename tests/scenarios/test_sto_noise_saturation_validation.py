from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.scenarios.sto_noise_saturation_validation import (
    run_phase_2_5,
    saturation_epsilon_sweep,
    sto_noise_saturation_validation,
)
from risk_aware_active_suspension.utils.config import ObserverParams


def test_saturation_reduces_noisy_sto_chatter(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    params = ObserverParams(lambda_1=100.0, lambda_2=2000.0, epsilon=0.02)

    result = sto_noise_saturation_validation(plant, params, duration=4.0)

    assert result["chatter_reduction_ratio"] < 0.7
    assert result["saturated_correlation_after_transient"] > 0.8
    assert result["saturated_rmse_ratio_after_transient"] < 0.2


def test_epsilon_sweep_shows_chatter_accuracy_tradeoff(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    params = ObserverParams(lambda_1=100.0, lambda_2=2000.0, epsilon=0.02)

    result = saturation_epsilon_sweep(plant, params)

    assert result["chatter_reduction_ratio"][0] > result["chatter_reduction_ratio"][-1]
    assert result["rmse_ratio"][0] < result["rmse_ratio"][-1]


def test_phase_2_5_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_2_5(
        vehicle_config_path=repo_root / "configs" / "vehicle_default.yaml",
        observer_config_path=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "epsilon_sweep.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "force_tracking_noise_saturation.png").exists()
    assert (run_dir / "figures" / "force_error_noise_saturation.png").exists()
    assert (run_dir / "figures" / "switching_signal.png").exists()
    assert (run_dir / "figures" / "velocity_noise.png").exists()
    assert (run_dir / "figures" / "epsilon_sweep.png").exists()
