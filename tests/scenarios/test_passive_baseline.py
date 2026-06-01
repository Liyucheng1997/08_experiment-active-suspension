from risk_aware_active_suspension.controllers.passive import PassiveController
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.passive_baseline import (
    passive_closed_loop,
    run_phase_3_1,
)


def test_short_passive_closed_loop_is_bit_identical_and_has_realistic_metrics(
    default_vehicle,
) -> None:
    plant = FullCar(default_vehicle)
    controller = PassiveController(n_actuators=4)

    result = passive_closed_loop(plant, controller, duration=8.0, dt=0.002)

    assert result["bit_identical_to_open_loop"] is True
    assert result["max_actuator_force"] == 0.0
    assert 0.3 <= result["vertical_accel_rms"] <= 0.6
    assert result["min_tire_force"] > 0.0


def test_phase_3_1_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_3_1(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "figures" / "vertical_acceleration.png").exists()
    assert (run_dir / "figures" / "actuator_forces.png").exists()
    assert (run_dir / "figures" / "tire_normal_forces.png").exists()

    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "bit_identical_to_open_loop,1" in metrics
    assert "max_actuator_force_n,0.0" in metrics
