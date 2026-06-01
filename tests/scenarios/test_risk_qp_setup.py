from risk_aware_active_suspension.scenarios.risk_qp_setup import (
    check_xi_zero_without_margin,
    compare_against_comfort_qp,
    run_phase_4_3,
)
from risk_aware_active_suspension.utils.config import lqr_from_yaml


def test_risk_qp_matches_comfort_qp_when_qp_zero(default_vehicle, repo_root) -> None:
    lqr = lqr_from_yaml(repo_root / "configs" / "controller_default.yaml")
    diffs = compare_against_comfort_qp(default_vehicle, lqr, n_samples=100)
    assert diffs["max_abs_diff_N"] < 1e-2


def test_xi_zero_without_margin(default_vehicle, repo_root) -> None:
    lqr = lqr_from_yaml(repo_root / "configs" / "controller_default.yaml")
    chk = check_xi_zero_without_margin(default_vehicle, lqr, n_samples=100)
    assert chk["max_xi"] < 1e-6


def test_phase_4_3_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_4_3(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        results_root=tmp_path,
        duration=4.0,
    )
    for fig in (
        "sigma_qc_trajectory.png",
        "rho_proxy.png",
        "actuator_forces.png",
        "xi_trajectory.png",
    ):
        assert (run_dir / "figures" / fig).exists(), fig
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_qp_eq_comfort_qp_when_qp_zero,1" in metrics
    assert "acceptance_xi_zero_without_margin,1" in metrics
