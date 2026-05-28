import numpy as np
import pytest

from risk_aware_active_suspension.inputs.road import rounded_bump
from risk_aware_active_suspension.plants.half_car import HalfCar
from risk_aware_active_suspension.scenarios.half_car_validation import run_phase_1_3


def test_symmetric_road_excites_heave_not_roll(default_vehicle) -> None:
    plant = HalfCar(default_vehicle)
    t = np.arange(0.0, 2.0, 0.001)
    road = rounded_bump(t, height=0.01, start=0.2, duration=0.08)
    states = plant.simulate(t, np.column_stack([road, road]))

    assert np.max(np.abs(states[:, 2])) < 1e-10
    assert np.max(np.abs(states[:, 0])) > 1e-5


def test_anti_symmetric_road_excites_roll_not_heave(default_vehicle) -> None:
    plant = HalfCar(default_vehicle)
    t = np.arange(0.0, 2.0, 0.001)
    road = rounded_bump(t, height=0.01, start=0.2, duration=0.08)
    states = plant.simulate(t, np.column_stack([road, -road]))

    assert np.max(np.abs(states[:, 0])) < 1e-10
    assert np.max(np.abs(states[:, 2])) > 1e-5


def test_roll_natural_frequency_matches_analytical(default_vehicle) -> None:
    plant = HalfCar(default_vehicle)
    eigvals = np.linalg.eigvals(plant.A)
    roll_target = plant.analytical_roll_natural_frequency()
    modal_freqs = np.array(sorted(abs(ev) for ev in eigvals if ev.imag > 0.0))

    assert np.min(np.abs(modal_freqs - roll_target)) / roll_target < 0.05


def test_half_car_rejects_bad_road_shape(default_vehicle) -> None:
    plant = HalfCar(default_vehicle)

    with pytest.raises(ValueError, match="w_seq"):
        plant.simulate([0.0, 0.1], [[0.0, 0.0, 0.0]])


def test_phase_1_3_runner_writes_acceptance_artifacts(tmp_path, repo_root) -> None:
    run_dir = run_phase_1_3(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "symmetric_decoupling.png").exists()
    assert (run_dir / "figures" / "anti_symmetric_decoupling.png").exists()
