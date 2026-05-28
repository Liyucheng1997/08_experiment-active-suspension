from pathlib import Path

import numpy as np

from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.scenarios.quarter_car_validation import (
    bump_validation,
    chirp_validation,
    free_vibration_validation,
    frequency_sweep_validation,
    run_phase_1_2,
)


def test_free_vibration_estimates_modal_parameters(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)

    result = free_vibration_validation(plant)

    assert result["omega_n_rel_error"] < 0.02
    assert result["zeta_rel_error"] < 0.05


def test_frequency_sweep_finds_sprung_and_unsprung_peaks(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)

    result = frequency_sweep_validation(plant)

    assert 1.0 <= result["sprung_peak_hz"] <= 1.8
    assert 8.0 <= result["unsprung_peak_hz"] <= 12.5


def test_chirp_response_is_finite(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)

    result = chirp_validation(plant, duration=2.0)

    assert result["states"].shape == (len(result["t"]), 4)
    assert np.all(np.isfinite(result["sprung_accel"]))


def test_standard_bump_keeps_tire_force_positive(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)

    result = bump_validation(plant)

    assert result["min_tire_force"] > 0.0
    assert result["peak_sprung_accel"] > 0.0


def test_phase_1_2_runner_writes_acceptance_artifacts(tmp_path: Path, repo_root: Path) -> None:
    run_dir = run_phase_1_2(
        config_path=repo_root / "configs" / "vehicle_default.yaml",
        results_root=tmp_path,
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "log.txt").exists()
    assert (run_dir / "raw.npz").exists()
    assert (run_dir / "figures" / "free_vibration.png").exists()
    assert (run_dir / "figures" / "transmissibility.png").exists()
    assert (run_dir / "figures" / "bump_tire_force.png").exists()
    assert (run_dir / "figures" / "chirp_response.png").exists()
