from risk_aware_active_suspension.scenarios.lqr_baseline import (
    compare_full_car,
    compare_quarter_car,
    run_phase_3_3,
)
from risk_aware_active_suspension.utils.config import LQRParams, lqr_from_yaml


def test_quarter_car_lqr_beats_skyhook_by_10pct(default_vehicle, repo_root) -> None:
    lqr = lqr_from_yaml(repo_root / "configs" / "controller_default.yaml")
    r = compare_quarter_car(default_vehicle, lqr, c_sky=22000.0, duration=15.0)
    assert r["lqr_poles_max_abs"] < 1.0
    improvement = 100.0 * (r["skyhook_rms"] - r["lqr_rms"]) / r["skyhook_rms"]
    assert improvement >= 10.0


def test_full_car_lqr_beats_skyhook_on_heave_by_10pct(default_vehicle, repo_root) -> None:
    lqr = lqr_from_yaml(repo_root / "configs" / "controller_default.yaml")
    r = compare_full_car(default_vehicle, lqr, c_sky=22000.0, duration=15.0)
    assert r["lqr_poles_max_abs"] < 1.0
    heave_improve = (
        100.0 * (r["skyhook_rms"]["heave"] - r["lqr_rms"]["heave"]) / r["skyhook_rms"]["heave"]
    )
    assert heave_improve >= 10.0
    # All three axes should be at least as good as skyhook (the tuned weights
    # are the contract that 3.5 relies on).
    for axis in ("roll", "pitch"):
        improve = (
            100.0 * (r["skyhook_rms"][axis] - r["lqr_rms"][axis]) / r["skyhook_rms"][axis]
        )
        assert improve > 0.0, axis


def test_phase_3_3_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_3_3(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        results_root=tmp_path,
    )
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "controller_config.yaml").exists()
    for fig in (
        "quarter_accel_compare.png",
        "full_heave_compare.png",
        "full_lqr_actuator_forces.png",
    ):
        assert (run_dir / "figures" / fig).exists(), fig
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_poles_inside_unit_circle,1" in metrics
    assert "acceptance_full_heave_vs_skyhook_ge_10pct,1" in metrics
    assert "acceptance_quarter_vs_skyhook_ge_10pct,1" in metrics
