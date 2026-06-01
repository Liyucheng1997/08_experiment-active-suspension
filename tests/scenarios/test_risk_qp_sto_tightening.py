from risk_aware_active_suspension.scenarios.risk_qp_sto_tightening import run_phase_4_7


def test_phase_4_7_runner_writes_tightening_diagnostics(tmp_path, repo_root) -> None:
    run_dir = run_phase_4_7(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        observer_config=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "tightening_sweep.csv").exists()
    for fig in ("tightening_sweep.png", "fz_safe_overshoot.png"):
        assert (run_dir / "figures" / fig).exists(), fig

    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_scenario_engages_constraint,1" in metrics
    assert "acceptance_safe_estimate_more_conservative_at_configured,1" in metrics
    assert "acceptance_xi_grows_with_tightening,1" in metrics
    assert "acceptance_all_runs_stable,1" in metrics
