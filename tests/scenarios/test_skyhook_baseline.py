import numpy as np

from risk_aware_active_suspension.scenarios.skyhook_baseline import (
    passive_vs_skyhook_full,
    passive_vs_skyhook_quarter,
    run_phase_3_2,
    tune_quarter_car_skyhook,
)


def test_quarter_car_skyhook_beats_passive_by_at_least_20pct(default_vehicle) -> None:
    result = passive_vs_skyhook_quarter(default_vehicle, c_sky=15000.0, duration=20.0)
    assert result["improvement_pct"] >= 20.0
    assert result["skyhook_accel_rms"] < result["passive_accel_rms"]


def test_full_car_skyhook_reduces_all_three_axes(default_vehicle) -> None:
    result = passive_vs_skyhook_full(default_vehicle, c_sky=15000.0, duration=20.0)
    for axis in ("heave", "roll", "pitch"):
        assert result["improvement_pct"][axis] > 0.0, axis


def test_tuner_selects_a_feasible_c_sky(default_vehicle) -> None:
    tuning = tune_quarter_car_skyhook(default_vehicle, duration=10.0)
    assert tuning["best_c_sky"] > 0.0
    assert tuning["best_improvement_pct"] > 0.0
    # All grid rows reported
    assert all("c_sky" in row and "improvement_pct" in row for row in tuning["sweep"])


def test_phase_3_2_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_3_2(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "config.yaml").exists()
    for fig in (
        "quarter_accel_compare.png",
        "quarter_tuning_sweep.png",
        "full_heave_compare.png",
        "full_actuator_forces.png",
    ):
        assert (run_dir / "figures" / fig).exists(), fig
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_quarter_ge_20pct,1" in metrics
    assert "acceptance_full_all_axes_improved,1" in metrics
