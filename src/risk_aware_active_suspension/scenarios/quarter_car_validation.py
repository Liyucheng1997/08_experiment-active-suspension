from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.signal import chirp, find_peaks

from risk_aware_active_suspension.inputs.road import rounded_bump
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.utils.config import VehicleParams, from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_freq_response, plot_side_by_side, plot_timeseries


def free_vibration_validation(
    plant: QuarterCar,
    duration: float = 8.0,
    dt: float = 0.001,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration + dt, dt)
    states = plant.simulate(t, u_seq=0.0, w_seq=0.0, x0=[0.05, 0.0, 0.0, 0.0])
    z_s = states[:, 0]
    peak_indices, _ = find_peaks(z_s, prominence=1e-5)
    peak_indices = peak_indices[z_s[peak_indices] > 0.0]
    if len(peak_indices) < 3:
        raise RuntimeError("Not enough free-vibration peaks to estimate modal parameters.")

    peak_times = t[peak_indices[:4]]
    peak_values = z_s[peak_indices[:4]]
    period = float(np.mean(np.diff(peak_times)))
    omega_d = 2.0 * np.pi / period
    log_decrement = float(np.mean(np.log(peak_values[:-1] / peak_values[1:])))
    zeta = log_decrement / np.sqrt((2.0 * np.pi) ** 2 + log_decrement**2)
    omega_n = omega_d / np.sqrt(1.0 - zeta**2)
    expected_omega_n, expected_zeta = _sprung_mode_from_eigenvalues(plant)
    simple_omega_n = np.sqrt(plant.params.k_s / plant.m_s)
    simple_zeta = plant.params.c_s / (2.0 * np.sqrt(plant.params.k_s * plant.m_s))

    return {
        "t": t,
        "states": states,
        "omega_n": float(omega_n),
        "expected_omega_n": float(expected_omega_n),
        "simple_omega_n": float(simple_omega_n),
        "zeta": float(zeta),
        "expected_zeta": float(expected_zeta),
        "simple_zeta": float(simple_zeta),
        "omega_n_rel_error": abs(omega_n - expected_omega_n) / expected_omega_n,
        "zeta_rel_error": abs(zeta - expected_zeta) / expected_zeta,
        "simple_omega_n_rel_error": abs(omega_n - simple_omega_n) / simple_omega_n,
    }


def frequency_sweep_validation(
    plant: QuarterCar,
    f_min: float = 0.5,
    f_max: float = 25.0,
    n_points: int = 600,
) -> dict[str, np.ndarray | float]:
    freq_hz = np.geomspace(f_min, f_max, n_points)
    mag = np.array([_road_to_sprung_acceleration_mag(plant, freq) for freq in freq_hz])
    peaks, _ = find_peaks(mag, prominence=0.05)
    if len(peaks) < 2:
        raise RuntimeError("Expected sprung and unsprung transmissibility peaks.")

    ordered = peaks[np.argsort(mag[peaks])[-2:]]
    peak_freqs = np.sort(freq_hz[ordered])
    return {
        "freq_hz": freq_hz,
        "magnitude": mag,
        "sprung_peak_hz": float(peak_freqs[0]),
        "unsprung_peak_hz": float(peak_freqs[1]),
    }


def chirp_validation(
    plant: QuarterCar,
    duration: float = 20.0,
    dt: float = 0.001,
    amplitude: float = 0.005,
) -> dict[str, np.ndarray]:
    t = np.arange(0.0, duration + dt, dt)
    road = amplitude * chirp(t, f0=0.5, f1=25.0, t1=duration, method="logarithmic")
    states = plant.simulate(t, u_seq=0.0, w_seq=road)
    sprung_accel = np.array([plant.sprung_acceleration(x, w=w) for x, w in zip(states, road)])
    return {"t": t, "road": road, "states": states, "sprung_accel": sprung_accel}


def bump_validation(
    plant: QuarterCar,
    duration: float = 2.0,
    dt: float = 0.0005,
    height: float = 0.01,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration + dt, dt)
    road = rounded_bump(t, height=height, start=0.4, duration=0.08)
    states = plant.simulate(t, u_seq=0.0, w_seq=road)
    tire_force = np.array([plant.tire_normal_force(x, w=w) for x, w in zip(states, road)])
    sprung_accel = np.array([plant.sprung_acceleration(x, w=w) for x, w in zip(states, road)])
    return {
        "t": t,
        "road": road,
        "states": states,
        "tire_force": tire_force,
        "sprung_accel": sprung_accel,
        "min_tire_force": float(np.min(tire_force)),
        "peak_sprung_accel": float(np.max(np.abs(sprung_accel))),
    }


def run_phase_1_2(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    config_path = Path(config_path)
    params = from_yaml(config_path)
    plant = QuarterCar(params)
    logger = RunLogger.create(results_root, phase="phase-1.2", step="quarter-car", descriptor="validation")
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    free = free_vibration_validation(plant)
    sweep = frequency_sweep_validation(plant)
    chirp_result = chirp_validation(plant)
    bump = bump_validation(plant)

    metrics = {
        "omega_n_rad_s": free["omega_n"],
        "omega_n_expected_rad_s": free["expected_omega_n"],
        "omega_n_simple_sqrt_ks_ms_rad_s": free["simple_omega_n"],
        "omega_n_rel_error": free["omega_n_rel_error"],
        "omega_n_simple_rel_error": free["simple_omega_n_rel_error"],
        "zeta": free["zeta"],
        "zeta_expected": free["expected_zeta"],
        "zeta_simple": free["simple_zeta"],
        "zeta_rel_error": free["zeta_rel_error"],
        "sprung_peak_hz": sweep["sprung_peak_hz"],
        "unsprung_peak_hz": sweep["unsprung_peak_hz"],
        "min_tire_force_n": bump["min_tire_force"],
        "peak_sprung_accel_m_s2": bump["peak_sprung_accel"],
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    plot_timeseries(
        free["t"],
        free["states"][:, 0],
        ylabel="Sprung displacement [m]",
        title="Quarter-car free vibration",
        save_path=logger.run_dir / "figures" / "free_vibration.png",
    )
    plot_freq_response(
        sweep["freq_hz"],
        sweep["magnitude"],
        ylabel="|ddot z_s / z_r| [1/s^2]",
        title="Road-to-sprung acceleration transmissibility",
        save_path=logger.run_dir / "figures" / "transmissibility.png",
    )
    plot_side_by_side(
        bump["t"],
        bump["road"],
        bump["tire_force"],
        left_label="Road bump [m]",
        right_label="Tire normal force [N]",
        title="Quarter-car rounded bump",
        save_path=logger.run_dir / "figures" / "bump_tire_force.png",
    )
    plot_side_by_side(
        chirp_result["t"],
        chirp_result["road"],
        chirp_result["sprung_accel"],
        left_label="Chirp road [m]",
        right_label="Sprung acceleration [m/s^2]",
        title="Quarter-car chirp response",
        save_path=logger.run_dir / "figures" / "chirp_response.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        free_t=free["t"],
        free_states=free["states"],
        sweep_freq_hz=sweep["freq_hz"],
        sweep_magnitude=sweep["magnitude"],
        bump_t=bump["t"],
        bump_road=bump["road"],
        bump_states=bump["states"],
        bump_tire_force=bump["tire_force"],
        chirp_t=chirp_result["t"],
        chirp_road=chirp_result["road"],
        chirp_states=chirp_result["states"],
        chirp_sprung_accel=chirp_result["sprung_accel"],
    )
    logger.log("Phase 1.2 quarter-car validation completed.")
    return logger.run_dir


def _road_to_sprung_acceleration_mag(plant: QuarterCar, freq_hz: float) -> float:
    jw = 1j * 2.0 * np.pi * freq_hz
    c = np.array([[0.0, 1.0, 0.0, 0.0]])
    h = c @ np.linalg.solve(jw * np.eye(4) - plant.A, plant.E)
    return float(abs(jw * h[0, 0]))


def _sprung_mode_from_eigenvalues(plant: QuarterCar) -> tuple[float, float]:
    eigvals = [ev for ev in np.linalg.eigvals(plant.A) if ev.imag > 0.0]
    sprung = min(eigvals, key=lambda ev: abs(ev))
    omega_n = abs(sprung)
    zeta = -sprung.real / omega_n
    return float(omega_n), float(zeta)


def main() -> None:
    run_dir = run_phase_1_2()
    print(run_dir)


if __name__ == "__main__":
    main()
