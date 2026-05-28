import csv

from risk_aware_active_suspension.scenarios.sto_gain_sweep_validation import (
    gain_sweep_validation,
    run_phase_2_3,
)


def test_gain_sweep_finds_feasible_default_region(default_vehicle) -> None:
    result = gain_sweep_validation(default_vehicle.m_u)

    assert result["feasible_count"] > 0
    lambda_1_values = list(result["lambda_1_values"])
    lambda_2_values = list(result["lambda_2_values"])
    default_i = lambda_1_values.index(20.0)
    default_j = lambda_2_values.index(75.0)
    assert result["constant_settling_s"][default_i, default_j] < 0.1
    assert result["constant_rmse_ratio"][default_i, default_j] < 0.01
    assert result["sinusoidal_rmse_ratio"][default_i, default_j] < 0.05


def test_phase_2_3_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_2_3(
        vehicle_config_path=repo_root / "configs" / "vehicle_default.yaml",
        observer_config_path=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "gain_sweep.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "sinusoidal_rmse_heatmap.png").exists()
    assert (run_dir / "figures" / "constant_settling_heatmap.png").exists()
    assert (run_dir / "figures" / "feasible_region.png").exists()
    with (run_dir / "gain_sweep.csv").open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert len(rows) > 1
