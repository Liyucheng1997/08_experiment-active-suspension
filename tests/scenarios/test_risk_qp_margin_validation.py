from risk_aware_active_suspension.scenarios.risk_qp_margin_validation import run_phase_4_4


def test_phase_4_4_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_4_4(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        results_root=tmp_path,
    )
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    for fig in (
        "caseA_force_balance.png",
        "caseA_rho.png",
        "caseB_xi_and_force.png",
        "caseB_force_balance.png",
    ):
        assert (run_dir / "figures" / fig).exists(), fig

    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_caseA_authority_sufficient_protects_margin,1" in metrics
    assert "acceptance_caseB_authority_insufficient_xi_positive,1" in metrics
