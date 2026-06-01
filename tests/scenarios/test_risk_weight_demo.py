from risk_aware_active_suspension.scenarios.risk_weight_demo import run_phase_4_2


def test_phase_4_2_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_4_2(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    for fig in ("sigma_vs_rho.png", "qc_qp_vs_rho.png", "qp_local_vs_rho_ij.png"):
        assert (run_dir / "figures" / fig).exists(), fig
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "sigma_at_rho_th,0.5" in metrics
    # κ_ρ=0 collapse: uniform global weight has zero std.
    assert "kappa_zero_uniform_std,0.0" in metrics
