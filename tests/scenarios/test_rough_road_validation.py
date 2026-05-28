from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.rough_road_validation import (
    rough_road_validation,
    run_phase_1_6,
)


def test_short_rough_road_run_has_realistic_signal_bounds(default_vehicle) -> None:
    plant = FullCar(default_vehicle)

    result = rough_road_validation(plant, duration=8.0, dt=0.002)

    assert 0.3 <= result["vertical_accel_rms"] <= 0.6
    assert result["stroke_peak_abs"] < 0.12
    assert result["min_tire_force"] > 0.0


def test_phase_1_6_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_1_6(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "vertical_acceleration.png").exists()
    assert (run_dir / "figures" / "suspension_strokes.png").exists()
    assert (run_dir / "figures" / "tire_normal_forces.png").exists()
    assert (run_dir / "figures" / "road_inputs.png").exists()
