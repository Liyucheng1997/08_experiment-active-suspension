from risk_aware_active_suspension.scenarios.comfort_qp_baseline import (
    compare_unconstrained_match,
    run_phase_3_4,
)
from risk_aware_active_suspension.utils.config import lqr_from_yaml


def test_unconstrained_match_within_tolerance(default_vehicle, repo_root) -> None:
    lqr = lqr_from_yaml(repo_root / "configs" / "controller_default.yaml")
    diffs = compare_unconstrained_match(default_vehicle, lqr, n_samples=100)
    assert diffs["quarter_max_abs_diff_N"] < 1e-3
    assert diffs["full_max_abs_diff_N"] < 1e-3


def test_phase_3_4_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_3_4(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        results_root=tmp_path,
        duration=4.0,
    )
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    for fig in (
        "full_heave_qp_vs_lqr.png",
        "full_qp_actuator_forces.png",
        "full_qp_solve_times.png",
    ):
        assert (run_dir / "figures" / fig).exists(), fig
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_unconstrained_matches_lqr,1" in metrics
    assert "acceptance_solve_time_p95_lt_1ms,1" in metrics
