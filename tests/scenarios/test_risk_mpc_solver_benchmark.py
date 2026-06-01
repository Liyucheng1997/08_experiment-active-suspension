from risk_aware_active_suspension.scenarios.risk_mpc_solver_benchmark import run_phase_5_4


def test_phase_5_4_runner_writes_solver_benchmark_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_5_4(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        observer_config=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
        duration=1.4,
        n_benchmark_solves=8,
    )

    assert (run_dir / "solver_benchmark.csv").exists()
    assert (run_dir / "figures" / "solver_timing_by_horizon.png").exists()
    assert (run_dir / "figures" / "solver_iterations_by_horizon.png").exists()
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_N10_p95_setup_solve_within_Ts" in metrics
    assert "largest_feasible_horizon_p95_wall_within_Ts" in metrics
