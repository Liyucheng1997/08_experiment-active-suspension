from risk_aware_active_suspension.scenarios.risk_mpc_vs_qp_comparison import run_phase_5_5


def test_phase_5_5_runner_writes_mpc_vs_qp_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_5_5(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        observer_config=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
        duration=2.3,
    )

    expected_figures = [
        "rho_max_mpc_vs_qp.png",
        "outer_front_rho_mpc_vs_qp.png",
        "heave_mpc_vs_qp.png",
        "forces_mpc_vs_qp.png",
    ]
    for figure_name in expected_figures:
        assert (run_dir / "figures" / figure_name).exists()
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_mpc_extra_rho_reduction_ge_5pct,1" in metrics
    assert "acceptance_mpc_no_force_saturation" in metrics
