from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.scenarios.sto_zero_disturbance_validation import (
    run_phase_2_1,
    sto_zero_disturbance_validation,
)
from risk_aware_active_suspension.utils.config import ObserverParams


def test_sto_zero_disturbance_has_no_chi_drift(default_vehicle) -> None:
    sto = STO(ObserverParams(lambda_1=20.0, lambda_2=50.0), unsprung_mass=default_vehicle.m_u)

    result = sto_zero_disturbance_validation(sto)

    assert result["max_abs_chi_after_transient"] < 1e-6
    assert result["max_abs_dz_after_transient"] < 1e-6
    assert result["max_abs_velocity_error_after_transient"] < 0.002


def test_phase_2_1_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_2_1(
        vehicle_config_path=repo_root / "configs" / "vehicle_default.yaml",
        observer_config_path=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "velocity_tracking.png").exists()
    assert (run_dir / "figures" / "chi_hat.png").exists()
    assert (run_dir / "figures" / "d_z_hat.png").exists()
