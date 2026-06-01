from risk_aware_active_suspension.scenarios.risk_qp_fullcar_ground_truth import run_phase_4_5


def test_phase_4_5_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_4_5(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    for fig in (
        "rho_max_physical.png",
        "rho_outer_front.png",
        "risk_qp_forces.png",
        "heave_accel_tradeoff.png",
        "outer_front_margin.png",
    ):
        assert (run_dir / "figures" / fig).exists(), fig

    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_scenario_engages_constraint,1" in metrics
    assert "acceptance_peak_rho_reduction_ge_10pct," in metrics
    assert "acceptance_comfort_degradation_lt_30pct," in metrics
