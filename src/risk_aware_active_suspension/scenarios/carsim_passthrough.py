from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.plants.carsim_fmu import CarSimFmuPlant
from risk_aware_active_suspension.utils.logger import RunLogger


DEFAULT_FMU_PATH = Path(
    r"C:\Users\Public\Documents\CarSim2019.1_Data\Extensions\FMU_FMI\event_FMI2.fmu"
)


def run_phase_7_1_passthrough(
    fmu_path: str | Path = DEFAULT_FMU_PATH,
    results_root: str | Path = "results",
    duration: float = 2.0,
    sample_time: float = 0.005,
    speed: float = 80.0,
    mu: float = 0.9,
) -> Path:
    """Run the Phase 7.1 zero-force FMU passthrough check."""

    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive.")

    plant = CarSimFmuPlant(Path(fmu_path))
    logger = RunLogger.create(
        results_root,
        phase="phase-7.1",
        step="carsim-fmu",
        descriptor="passthrough",
    )
    t = np.arange(0.0, duration + 0.5 * sample_time, sample_time)
    result = plant.simulate(
        t,
        speed=speed,
        steer=0.0,
        active_force=0.0,
        brake_pressure=0.0,
        road_height=0.0,
        mu_x=mu,
        mu_y=mu,
        output_interval=sample_time,
    )
    fz = plant.collect_corners(result, plant.signal_map.fz)
    jounce = plant.collect_corners(result, plant.signal_map.jounce)
    alpha = plant.collect_corners(result, plant.signal_map.alpha)
    kappa = plant.collect_corners(result, plant.signal_map.kappa)

    valid = result["time"] >= min(1.0, 0.25 * duration)
    speed_error = np.asarray(result["Vx"], dtype=float) - speed
    metrics = {
        "duration_s": duration,
        "sample_time_s": sample_time,
        "speed_command": speed,
        "vx_mean": float(np.mean(result["Vx"])),
        "vx_mean_after_initial": float(np.mean(result["Vx"][valid])),
        "vx_final": float(result["Vx"][-1]),
        "vx_error_final": float(speed_error[-1]),
        "fz_min_after_initial_n": float(np.min(fz[valid])),
        "fz_max_after_initial_n": float(np.max(fz[valid])),
        "fz_front_mean_after_initial_n": float(np.mean(fz[valid, :2])),
        "fz_rear_mean_after_initial_n": float(np.mean(fz[valid, 2:])),
        "max_abs_roll_after_initial_deg": float(np.max(np.abs(result["Roll"][valid]))),
        "max_abs_pitch_after_initial_deg": float(np.max(np.abs(result["Pitch"][valid]))),
        "max_abs_az_sm_after_initial": float(np.max(np.abs(result["Az_SM"][valid]))),
        "max_abs_jounce_after_initial_mm": float(np.max(np.abs(jounce[valid]))),
        "max_abs_alpha_after_initial_deg": float(np.max(np.abs(alpha[valid]))),
        "max_abs_kappa_after_initial": float(np.max(np.abs(kappa[valid]))),
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)
    logger.log("Phase 7.1 CarSim FMU passthrough completed.")

    np.savez(
        logger.run_dir / "raw.npz",
        **{name: values for name, values in result.items()},
        fz_FL_FR_RL_RR=fz,
        jounce_FL_FR_RL_RR=jounce,
        alpha_FL_FR_RL_RR=alpha,
        kappa_FL_FR_RL_RR=kappa,
    )
    _plot_passthrough(logger.run_dir / "figures" / "passthrough_overview.png", result, fz)
    return logger.run_dir


def _plot_passthrough(path: Path, result: dict[str, np.ndarray], fz: np.ndarray) -> None:
    t = result["time"]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.0), sharex=True)
    axes[0].plot(t, result["Vx"])
    axes[0].set_ylabel("Vx")
    axes[0].set_title("CarSim FMU passthrough")
    axes[1].plot(t, result["Az_SM"])
    axes[1].set_ylabel("Az_SM")
    axes[2].plot(t, fz)
    axes[2].set_ylabel("Fz [N]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(["FL", "FR", "RL", "RR"], ncol=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    print(run_phase_7_1_passthrough())
