from risk_aware_active_suspension.scenarios.tire_metric_demo import run_phase_4_1


def test_phase_4_1_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_4_1(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "rho_vs_fz.png").exists()
    assert (run_dir / "figures" / "fz_required_vs_fc.png").exists()
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "f_z_required_at_zero_demand_n,0.0" in metrics
    # Inverse identity within numerical noise (string match on prefix).
    assert "rho_at_inverse_check,0.85" in metrics
