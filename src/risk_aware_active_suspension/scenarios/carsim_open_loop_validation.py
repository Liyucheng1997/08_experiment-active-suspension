from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from risk_aware_active_suspension.plants.carsim_fmu import CarSimFmuPlant
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.carsim_passthrough import DEFAULT_FMU_PATH
from risk_aware_active_suspension.utils.config import from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


G = 9.81
CORNER_LABELS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class OpenLoopCase:
    name: str
    steer_deg: float = 0.0
    brake_mpa: float = 0.0


CASES = (
    OpenLoopCase("straight_smooth_80"),
    OpenLoopCase("brake_step_1mpa", brake_mpa=1.0),
    OpenLoopCase("steer_step_20deg", steer_deg=20.0),
)


def run_phase_7_2_open_loop_validation(
    fmu_path: str | Path = DEFAULT_FMU_PATH,
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
    duration: float = 4.0,
    sample_time: float = 0.005,
    speed: float = 80.0,
    mu: float = 0.9,
    step_start: float = 1.0,
) -> Path:
    """Run Phase 7.2 open-loop CarSim vs analytical full-car checks."""

    vehicle_config = Path(vehicle_config)
    carsim = CarSimFmuPlant(Path(fmu_path))
    full_car = FullCar(from_yaml(vehicle_config))
    logger = RunLogger.create(
        results_root,
        phase="phase-7.2",
        step="carsim",
        descriptor="open-loop-validation",
    )
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")

    t = np.arange(0.0, duration + 0.5 * sample_time, sample_time)
    steady = t >= max(0.1, step_start + 1.0, 0.75 * duration)
    runs: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | str]] = []

    for case in CASES:
        steer = np.zeros_like(t)
        brake = np.zeros((len(t), 4), dtype=float)
        if case.steer_deg:
            steer[t >= step_start] = case.steer_deg
        if case.brake_mpa:
            brake[t >= step_start, :] = case.brake_mpa
        sim = _run_carsim(carsim, t, speed, steer, brake, mu, sample_time)
        analytical = _run_analytical(full_car, sim, sample_time)
        runs[case.name] = {**sim, **{f"analytical_{k}": v for k, v in analytical.items()}}
        rows.extend(_case_rows(case.name, sim, analytical, steady))

    baseline = runs["straight_smooth_80"]
    for case in CASES[1:]:
        rows.extend(_delta_rows(case.name, baseline, runs[case.name], steady))

    _write_comparison(logger.run_dir / "open_loop_comparison.csv", rows)
    _log_summary(logger, rows, runs, t, steady, sample_time)
    _save_raw(logger.run_dir / "raw.npz", runs)
    for case in CASES:
        _plot_case(logger.run_dir / "figures" / f"{case.name}.png", case.name, t, runs[case.name])
    logger.log("Phase 7.2 CarSim open-loop validation completed.")
    return logger.run_dir


def _run_carsim(
    plant: CarSimFmuPlant,
    t: np.ndarray,
    speed: float,
    steer: np.ndarray,
    brake: np.ndarray,
    mu: float,
    sample_time: float,
) -> dict[str, np.ndarray]:
    outputs = (
        "Vx",
        "Ax",
        "Ay",
        "Az_SM",
        "Zcg_SM",
        "Roll",
        "Pitch",
        "Fz_L1",
        "Fz_R1",
        "Fz_L2",
        "Fz_R2",
        "Alpha_L1",
        "Alpha_R1",
        "Alpha_L2",
        "Alpha_R2",
        "Kappa_L1",
        "Kappa_R1",
        "Kappa_L2",
        "Kappa_R2",
    )
    result = plant.simulate(
        t,
        speed=speed,
        steer=steer,
        active_force=0.0,
        brake_pressure=brake,
        road_height=0.0,
        mu_x=mu,
        mu_y=mu,
        output_interval=sample_time,
        outputs=outputs,
    )
    fz = plant.collect_corners(result, plant.signal_map.fz)
    alpha = plant.collect_corners(result, plant.signal_map.alpha)
    kappa = plant.collect_corners(result, plant.signal_map.kappa)
    return {
        "time": result["time"],
        "vx_kmh": result["Vx"],
        "ax_g": result["Ax"],
        "ay_g": result["Ay"],
        "az_sm_g": result["Az_SM"],
        "z_sm_m": result["Zcg_SM"],
        "roll_deg": result["Roll"],
        "pitch_deg": result["Pitch"],
        "fz_n": fz,
        "alpha_deg": alpha,
        "kappa": kappa,
    }


def _run_analytical(plant: FullCar, carsim_run: dict[str, np.ndarray], sample_time: float) -> dict[str, np.ndarray]:
    t = carsim_run["time"]
    roads = np.zeros((len(t), 4), dtype=float)
    states = np.zeros((len(t), 14), dtype=float)
    fz = np.zeros((len(t), 4), dtype=float)
    body_accel = np.zeros((len(t), 3), dtype=float)
    for idx in range(len(t) - 1):
        # The analytical full-car body_acc convention uses positive a_x for
        # front-axle unloading. CarSim's exported Ax has the opposite sign for
        # the load-transfer effect in this FMU, so flip only the longitudinal
        # channel at the interface.
        ax_ay = np.array([-carsim_run["ax_g"][idx] * G, carsim_run["ay_g"][idx] * G])
        fz[idx] = plant.tire_normal_forces(states[idx], roads[idx])
        body_accel[idx] = plant.body_accelerations(states[idx], roads[idx], body_acc=ax_ay)
        states[idx + 1] = plant.step(
            states[idx],
            roads[idx],
            sample_time,
            body_acc=ax_ay,
        )
    final_ax_ay = np.array([-carsim_run["ax_g"][-1] * G, carsim_run["ay_g"][-1] * G])
    fz[-1] = plant.tire_normal_forces(states[-1], roads[-1])
    body_accel[-1] = plant.body_accelerations(states[-1], roads[-1], body_acc=final_ax_ay)
    return {
        "states": states,
        "fz_n": fz,
        "az_sm_g": body_accel[:, 0] / G,
        "roll_deg": np.rad2deg(states[:, 2]),
        "pitch_deg": np.rad2deg(states[:, 4]),
    }


def _case_rows(
    case_name: str,
    carsim: dict[str, np.ndarray],
    analytical: dict[str, np.ndarray],
    valid: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    carsim_mean = np.mean(carsim["fz_n"][valid], axis=0)
    analytical_mean = np.mean(analytical["fz_n"][valid], axis=0)
    for idx, corner in enumerate(CORNER_LABELS):
        rows.append(
            {
                "scenario": case_name,
                "metric": "steady_fz",
                "corner": corner,
                "carsim": float(carsim_mean[idx]),
                "analytical": float(analytical_mean[idx]),
                "abs_error": float(abs(carsim_mean[idx] - analytical_mean[idx])),
                "rel_error": float(abs(carsim_mean[idx] - analytical_mean[idx]) / max(abs(carsim_mean[idx]), 1.0)),
            }
        )
    return rows


def _delta_rows(
    case_name: str,
    baseline: dict[str, np.ndarray],
    case: dict[str, np.ndarray],
    valid: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    carsim_delta = np.mean(case["fz_n"][valid], axis=0) - np.mean(baseline["fz_n"][valid], axis=0)
    analytical_delta = np.mean(case["analytical_fz_n"][valid], axis=0) - np.mean(
        baseline["analytical_fz_n"][valid], axis=0
    )
    for idx, corner in enumerate(CORNER_LABELS):
        denom = max(abs(carsim_delta[idx]), 1.0)
        rows.append(
            {
                "scenario": case_name,
                "metric": "steady_delta_fz",
                "corner": corner,
                "carsim": float(carsim_delta[idx]),
                "analytical": float(analytical_delta[idx]),
                "abs_error": float(abs(carsim_delta[idx] - analytical_delta[idx])),
                "rel_error": float(abs(carsim_delta[idx] - analytical_delta[idx]) / denom),
            }
        )
    return rows


def _write_comparison(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["scenario", "metric", "corner", "carsim", "analytical", "abs_error", "rel_error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _log_summary(
    logger: RunLogger,
    rows: list[dict[str, float | str]],
    runs: dict[str, dict[str, np.ndarray]],
    t: np.ndarray,
    valid: np.ndarray,
    sample_time: float,
) -> None:
    steady_rows = [r for r in rows if r["metric"] == "steady_fz"]
    delta_rows = [r for r in rows if r["metric"] == "steady_delta_fz"]
    max_steady_rel_error = max(float(r["rel_error"]) for r in steady_rows)
    max_delta_abs_error = max(float(r["abs_error"]) for r in delta_rows)
    max_delta_rel_error = max(float(r["rel_error"]) for r in delta_rows)
    straight = runs["straight_smooth_80"]
    band_power_carsim = _band_power(straight["az_sm_g"][valid], sample_time, 0.1, 10.0)
    band_power_analytical = _band_power(straight["analytical_az_sm_g"][valid], sample_time, 0.1, 10.0)
    logger.log_kv("max_steady_fz_rel_error", max_steady_rel_error)
    logger.log_kv("max_delta_fz_abs_error_n", max_delta_abs_error)
    logger.log_kv("max_delta_fz_rel_error", max_delta_rel_error)
    logger.log_kv("straight_az_psd_band_power_carsim_g2", band_power_carsim)
    logger.log_kv("straight_az_psd_band_power_analytical_g2", band_power_analytical)
    logger.log_kv("straight_vx_mean_kmh", float(np.mean(straight["vx_kmh"][valid])))
    logger.log_kv("acceptance_delta_fz_rel_error_lt_0p25", int(max_delta_rel_error < 0.25))
    logger.log_kv("acceptance_straight_speed_near_80", int(abs(np.mean(straight["vx_kmh"][valid]) - 80.0) < 1.0))
    logger.log(f"Validation window: {float(t[valid][0]):.3f}s to {float(t[valid][-1]):.3f}s")


def _band_power(signal: np.ndarray, sample_time: float, f_min: float, f_max: float) -> float:
    if len(signal) < 8:
        return 0.0
    fs = 1.0 / sample_time
    freq, psd = welch(signal - np.mean(signal), fs=fs, nperseg=min(512, len(signal)))
    mask = (freq >= f_min) & (freq <= f_max)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freq[mask]))


def _save_raw(path: Path, runs: dict[str, dict[str, np.ndarray]]) -> None:
    payload = {}
    for case_name, run in runs.items():
        for key, values in run.items():
            payload[f"{case_name}_{key}"] = values
    np.savez(path, **payload)


def _plot_case(path: Path, case_name: str, t: np.ndarray, run: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.5), sharex=True)
    axes[0].plot(t, run["vx_kmh"], label="CarSim Vx")
    axes[0].set_ylabel("Vx [km/h]")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, run["fz_n"])
    axes[1].plot(t, run["analytical_fz_n"], linestyle="--")
    axes[1].set_ylabel("Fz [N]")
    axes[1].legend([f"CS {c}" for c in CORNER_LABELS] + [f"AN {c}" for c in CORNER_LABELS], ncol=4, fontsize=7)
    axes[2].plot(t, run["az_sm_g"], label="CarSim Az_SM")
    axes[2].plot(t, run["analytical_az_sm_g"], linestyle="--", label="Analytical Az")
    axes[2].set_ylabel("Az [g]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(fontsize=8)
    fig.suptitle(case_name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    print(run_phase_7_2_open_loop_validation())


if __name__ == "__main__":
    main()
