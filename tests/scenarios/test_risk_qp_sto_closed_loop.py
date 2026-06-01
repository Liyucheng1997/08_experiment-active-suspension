from risk_aware_active_suspension.scenarios.risk_qp_sto_closed_loop import run_phase_4_6


def test_phase_4_6_runner_writes_sto_diagnostic_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_4_6(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        observer_config=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    for fig in (
        "rho_max_sto_vs_truth.png",
        "outer_front_fz_hat_vs_truth.png",
        "heave_accel_sto_vs_truth.png",
        "sto_qp_forces.png",
    ):
        assert (run_dir / "figures" / fig).exists(), fig

    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_fz_rmse_full_lt_500N,1" in metrics
    assert "acceptance_min_corner_correlation_gt_0p3,1" in metrics
    assert "acceptance_comfort_within_40pct_of_truth,1" in metrics
    assert "acceptance_no_nan,1" in metrics
