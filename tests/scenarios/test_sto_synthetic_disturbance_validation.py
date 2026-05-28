from risk_aware_active_suspension.scenarios.sto_synthetic_disturbance_validation import (
    constant_disturbance_validation,
    run_phase_2_2,
    sinusoidal_disturbance_validation,
)
from risk_aware_active_suspension.utils.config import ObserverParams


def test_constant_synthetic_disturbance_settles_fast(default_vehicle) -> None:
    params = ObserverParams(lambda_1=20.0, lambda_2=75.0)

    result = constant_disturbance_validation(default_vehicle.m_u, params)

    assert result["settling_time_s"] < 0.2
    assert result["steady_state_rmse_ratio"] < 0.01


def test_sinusoidal_synthetic_disturbance_tracks_after_transient(default_vehicle) -> None:
    params = ObserverParams(lambda_1=20.0, lambda_2=75.0)

    result = sinusoidal_disturbance_validation(default_vehicle.m_u, params)

    assert result["steady_state_rmse_ratio"] < 0.05


def test_phase_2_2_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_2_2(
        vehicle_config_path=repo_root / "configs" / "vehicle_default.yaml",
        observer_config_path=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "constant_disturbance.png").exists()
    assert (run_dir / "figures" / "sinusoidal_disturbance.png").exists()
    assert (run_dir / "figures" / "sinusoidal_error.png").exists()
