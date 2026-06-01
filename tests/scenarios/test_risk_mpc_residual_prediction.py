from risk_aware_active_suspension.scenarios.risk_mpc_residual_prediction import run_phase_5_3


def test_phase_5_3_runner_writes_residual_prediction_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_5_3(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        observer_config=repo_root / "configs" / "observer_default.yaml",
        results_root=tmp_path,
        duration=2.0,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    for fig in (
        "rho_max_residual_modes.png",
        "heave_residual_modes.png",
        "forces_residual_modes.png",
        "outer_front_fz_prediction_modes.png",
    ):
        assert (run_dir / "figures" / fig).exists(), fig

    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "freeze_stable_no_nan,1" in metrics
    assert "decay_alpha_0.95_stable_no_nan,1" in metrics
