import numpy as np
import pytest

from risk_aware_active_suspension.inputs.road import rounded_bump
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.scenarios.full_car_validation import (
    decoupling_validation,
    run_phase_1_4,
    static_load_validation,
)


def test_full_car_state_and_corner_conventions(default_vehicle) -> None:
    plant = FullCar(default_vehicle)

    assert CORNER_NAMES == ("FL", "FR", "RL", "RR")
    assert plant.A.shape == (14, 14)
    assert plant.E.shape == (14, 4)
    assert plant.corner_xy[0] == pytest.approx((default_vehicle.l_f, default_vehicle.t / 2.0))
    assert plant.corner_xy[-1] == pytest.approx((-default_vehicle.l_r, -default_vehicle.t / 2.0))


def test_static_loads_sum_to_weight_and_match_front_rear_split(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    result = static_load_validation(plant)

    assert result["total_load"] == pytest.approx(result["expected_total_load"], abs=1e-3)
    assert result["front_load"] == pytest.approx(result["expected_front_load"], abs=1e-3)
    assert result["rear_load"] == pytest.approx(result["expected_rear_load"], abs=1e-3)
    assert result["left_right_imbalance"] == pytest.approx(0.0, abs=1e-9)


def test_full_car_symmetric_road_excites_heave_only(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    t = np.arange(0.0, 4.0, 0.001)
    bump = rounded_bump(t, height=0.01, start=0.5, duration=1.0)
    states = plant.simulate(t, np.column_stack([bump, bump, bump, bump]))

    assert np.max(np.abs(states[:, 0])) > 1e-5
    assert np.max(np.abs(states[:, 2])) < 1e-10
    assert np.max(np.abs(states[:, 4])) / np.max(np.abs(states[:, 0])) < 0.01


def test_full_car_roll_and_pitch_decoupling(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    result = decoupling_validation(plant)

    assert result["roll_peak"] > 1e-5
    assert result["pitch_peak"] > 1e-5
    assert result["roll_heave_leakage"] < 1e-10
    assert result["roll_pitch_leakage"] < 1e-10
    assert result["pitch_heave_leakage"] / result["pitch_peak"] < 0.01
    assert result["pitch_roll_leakage"] < 1e-10


def test_tire_normal_forces_equal_static_loads_at_rest(default_vehicle) -> None:
    plant = FullCar(default_vehicle)

    forces = plant.tire_normal_forces(np.zeros(14), np.zeros(4))

    np.testing.assert_allclose(forces, plant.static_loads(), atol=1e-9)


def test_tire_normal_force_increases_on_positive_road_input(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    road = np.zeros(4)
    road[0] = 0.02

    forces = plant.tire_normal_forces(np.zeros(14), road)

    assert forces[0] == pytest.approx(plant.static_loads()[0] + default_vehicle.k_t * 0.02)


def test_tire_normal_forces_clamp_liftoff_to_contact_floor(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    state = np.zeros(14)
    state[6] = 1.0

    forces = plant.tire_normal_forces(state, np.zeros(4))

    assert forces[0] == 0.0


def test_full_car_signal_helpers_have_expected_shapes(default_vehicle) -> None:
    plant = FullCar(default_vehicle)
    states = np.zeros((3, 14))
    roads = np.zeros((3, 4))

    assert plant.batch_body_accelerations(states, roads).shape == (3, 3)
    assert plant.batch_suspension_strokes(states).shape == (3, 4)
    assert plant.batch_tire_normal_forces(states, roads).shape == (3, 4)


def test_full_car_rejects_bad_road_shape(default_vehicle) -> None:
    plant = FullCar(default_vehicle)

    with pytest.raises(ValueError, match="w_seq"):
        plant.simulate([0.0, 0.1], [[0.0, 0.0]])


def test_phase_1_4_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_1_4(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "heave_decoupling.png").exists()
    assert (run_dir / "figures" / "roll_decoupling.png").exists()
    assert (run_dir / "figures" / "pitch_decoupling.png").exists()
