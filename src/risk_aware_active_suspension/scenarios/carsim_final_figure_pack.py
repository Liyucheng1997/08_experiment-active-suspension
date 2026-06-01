from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.plants.full_car import CORNER_NAMES
from risk_aware_active_suspension.utils.logger import RunLogger


COLORS = {
    "comfort": "#C83737",
    "mpc": "#1F77B4",
    "safe": "#2B2B2B",
    "matched": "#0072B2",
    "mismatch": "#D55E00",
    "fz": "#009E73",
}


def run_phase_7_6_carsim_final_figures(
    results_root: str | Path = "results",
    phase74_source: str | Path | None = None,
    phase75_source: str | Path | None = None,
) -> Path:
    """Build the final CarSim figure/data pack from Phase 7.4 and 7.5 outputs."""

    _set_style()
    results_root = Path(results_root)
    phase74 = Path(phase74_source) if phase74_source is not None else _latest_run(results_root, "phase-7.4_carsim_risk-mpc-validation_")
    phase75 = Path(phase75_source) if phase75_source is not None else _latest_run(results_root, "phase-7.5_carsim_robustness-validation_")
    logger = RunLogger.create(results_root, phase="phase-7.6", step="carsim", descriptor="final-figure-pack")
    fig_dir = logger.run_dir / "figures"
    table_dir = logger.run_dir / "tables"
    data_dir = logger.run_dir / "data"
    fig_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    shutil.copy2(phase74 / "phase7_4_summary.csv", table_dir / "phase7_4_summary.csv")
    shutil.copy2(phase75 / "phase7_5_robustness_summary.csv", table_dir / "phase7_5_robustness_summary.csv")
    shutil.copy2(phase74 / "raw.npz", data_dir / "phase7_4_raw.npz")
    shutil.copy2(phase75 / "raw.npz", data_dir / "phase7_5_raw.npz")

    rows74 = _read_rows(table_dir / "phase7_4_summary.csv")
    rows75 = _read_rows(table_dir / "phase7_5_robustness_summary.csv")
    raw74 = np.load(data_dir / "phase7_4_raw.npz")
    raw75 = np.load(data_dir / "phase7_5_raw.npz")

    _write_key_metrics(table_dir / "phase7_key_metrics.csv", rows74, rows75)
    _plot_key_metrics(fig_dir / "fig01_carsim_key_metrics", rows74)
    _plot_s4_s8_rho_timeseries(fig_dir / "fig02_carsim_s4_s8_rho_timeseries", raw74)
    _plot_s8_load_and_force(fig_dir / "fig03_carsim_s8_load_force", raw74)
    _plot_robustness(fig_dir / "fig04_carsim_mu_mismatch", rows75, raw75)
    _plot_speed_and_accel(fig_dir / "fig05_carsim_s8_speed_accel", raw74)
    _write_report(logger.run_dir / "carsim_final_report.md", phase74, phase75, rows74, rows75)

    logger.log(f"Phase 7.4 source: {phase74}")
    logger.log(f"Phase 7.5 source: {phase75}")
    logger.log("Phase 7.6 CarSim final figure pack completed.")
    return logger.run_dir


def _latest_run(root: Path, prefix: str) -> Path:
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directory matching {prefix!r} under {root}.")
    return candidates[0]


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.7,
        }
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _save(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"))
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def _write_key_metrics(path: Path, rows74: list[dict[str, str]], rows75: list[dict[str, str]]) -> None:
    fields = [
        "group",
        "case",
        "plant_mu",
        "controller_mu",
        "comfort_peak_rho",
        "mpc_peak_rho",
        "rho_reduction_pct",
        "comfort_rho_p95",
        "mpc_rho_p95",
        "comfort_min_fz_n",
        "mpc_min_fz_n",
        "comfort_heave_rms_m_s2",
        "mpc_heave_rms_m_s2",
        "mpc_p95_solve_time_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group, rows in [("phase7_4", rows74), ("phase7_5", rows75)]:
            for row in rows:
                writer.writerow(
                    {
                        "group": group,
                        "case": row["scenario"],
                        "plant_mu": row["plant_mu"],
                        "controller_mu": row["controller_mu"],
                        "comfort_peak_rho": row["comfort_peak_rho"],
                        "mpc_peak_rho": row["mpc_peak_rho"],
                        "rho_reduction_pct": 100.0 * _f(row, "rho_reduction_ratio"),
                        "comfort_rho_p95": row["comfort_rho_p95"],
                        "mpc_rho_p95": row["mpc_rho_p95"],
                        "comfort_min_fz_n": row["comfort_min_fz_n"],
                        "mpc_min_fz_n": row["mpc_min_fz_n"],
                        "comfort_heave_rms_m_s2": row["comfort_heave_rms_m_s2"],
                        "mpc_heave_rms_m_s2": row["mpc_heave_rms_m_s2"],
                        "mpc_p95_solve_time_ms": row["mpc_p95_solve_time_ms"],
                    }
                )


def _plot_key_metrics(base_path: Path, rows: list[dict[str, str]]) -> None:
    labels = [row["scenario"].split("_", 1)[0] for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    comfort_peak = [_f(row, "comfort_peak_rho") for row in rows]
    mpc_peak = [_f(row, "mpc_peak_rho") for row in rows]
    comfort_p95 = [_f(row, "comfort_rho_p95") for row in rows]
    mpc_p95 = [_f(row, "mpc_rho_p95") for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, comfort, mpc, title, ylabel in [
        (axes[0], comfort_peak, mpc_peak, "Peak utilization", r"Peak $\rho_{max}$"),
        (axes[1], comfort_p95, mpc_p95, "95th percentile utilization", r"P95 $\rho_{max}$"),
    ]:
        ax.bar(x - width / 2, comfort, width, color=COLORS["comfort"], label="Comfort-QP")
        ax.bar(x + width / 2, mpc, width, color=COLORS["mpc"], label="Risk-MPC")
        ax.axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.0)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for idx, row in enumerate(rows):
            reduction = 100.0 * _f(row, "rho_reduction_ratio")
            ax.text(idx, max(comfort[idx], mpc[idx]) + 0.04, f"{reduction:.1f}%", ha="center", va="bottom", fontsize=8.5)
    axes[0].legend(loc="upper left")
    fig.suptitle("CarSim key scenarios: S4 stays mild, S8 gets protected")
    _save(fig, base_path)


def _plot_s4_s8_rho_timeseries(base_path: Path, raw: np.lib.npyio.NpzFile) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    for ax, scenario in zip(axes, ["S4_classC_cornering", "S8_worst_case"]):
        t = raw[f"{scenario}_comfort_qp_time"]
        comfort = np.max(raw[f"{scenario}_comfort_qp_rho_contact"], axis=1)
        mpc = np.max(raw[f"{scenario}_risk_mpc_rho_contact"], axis=1)
        ax.plot(t, comfort, color=COLORS["comfort"], label="Comfort-QP")
        ax.plot(t, mpc, color=COLORS["mpc"], label="Risk-MPC")
        ax.axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.0)
        ax.set_ylabel(r"$\rho_{max}$")
        ax.set_title(scenario)
        ax.set_xlim(0.5, float(t[-1]))
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("CarSim time history: friction utilization")
    _save(fig, base_path)


def _plot_s8_load_and_force(base_path: Path, raw: np.lib.npyio.NpzFile) -> None:
    scenario = "S8_worst_case"
    t = raw[f"{scenario}_comfort_qp_time"]
    comfort_fz = raw[f"{scenario}_comfort_qp_fz_true"]
    mpc_fz = raw[f"{scenario}_risk_mpc_fz_true"]
    force = raw[f"{scenario}_risk_mpc_forces"]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    for idx, corner in enumerate(CORNER_NAMES):
        axes[0].plot(t, comfort_fz[:, idx], linestyle="--", alpha=0.65, label=f"{corner} comfort")
        axes[0].plot(t, mpc_fz[:, idx], linewidth=1.6, label=f"{corner} MPC")
    axes[0].set_ylabel("Normal load [N]")
    axes[0].set_title("S8 CarSim tire normal load: MPC raises the weakest wheel load")
    axes[0].legend(ncol=4, fontsize=7.2)
    for idx, corner in enumerate(CORNER_NAMES):
        axes[1].plot(t, force[:, idx], label=corner)
    axes[1].set_ylabel("Controller force [N]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(ncol=4, loc="upper right")
    axes[1].set_xlim(0.5, float(t[-1]))
    _save(fig, base_path)


def _plot_robustness(base_path: Path, rows: list[dict[str, str]], raw: np.lib.npyio.NpzFile) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    labels = ["matched\n0.8/0.8", "mismatch\n0.6/0.8"]
    x = np.arange(len(rows))
    width = 0.34
    axes[0].bar(x - width / 2, [_f(row, "comfort_peak_rho") for row in rows], width, color=COLORS["comfort"], label="Comfort-QP")
    axes[0].bar(x + width / 2, [_f(row, "mpc_peak_rho") for row in rows], width, color=COLORS["mpc"], label="Risk-MPC")
    axes[0].axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.0)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel(r"Peak $\rho_{max}$")
    axes[0].set_title("μ mismatch robustness")
    axes[0].legend(loc="upper left")

    for case, color, label in [
        ("matched_mu_0p8", COLORS["matched"], "matched 0.8/0.8"),
        ("mismatch_actual_0p6_assumed_0p8", COLORS["mismatch"], "mismatch 0.6/0.8"),
    ]:
        t = raw[f"{case}_risk_mpc_time"]
        rho_trace = np.max(raw[f"{case}_risk_mpc_rho_contact"], axis=1)
        axes[1].plot(t, rho_trace, color=color, label=label)
    axes[1].axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.0)
    axes[1].set_xlim(0.5, 4.0)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"Risk-MPC $\rho_{max}$")
    axes[1].set_title("Risk-MPC degrades gracefully under μ mismatch")
    axes[1].legend(loc="upper right")
    _save(fig, base_path)


def _plot_speed_and_accel(base_path: Path, raw: np.lib.npyio.NpzFile) -> None:
    scenario = "S8_worst_case"
    t = raw[f"{scenario}_risk_mpc_time"]
    vx = raw[f"{scenario}_risk_mpc_vx_kmh"]
    ax_g = raw[f"{scenario}_risk_mpc_ax_g"]
    ay_g = raw[f"{scenario}_risk_mpc_ay_g"]
    az = raw[f"{scenario}_risk_mpc_az_m_s2"]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 5.8), sharex=True)
    axes[0].plot(t, vx, color="#4C4C4C")
    axes[0].set_ylabel("Vx [km/h]")
    axes[0].set_title("S8 CarSim vehicle states under Risk-MPC")
    axes[1].plot(t, ax_g, label="Ax", color="#0072B2")
    axes[1].plot(t, ay_g, label="Ay", color="#D55E00")
    axes[1].set_ylabel("Body accel. [g]")
    axes[1].legend(loc="upper right")
    axes[2].plot(t, az, color=COLORS["fz"])
    axes[2].set_ylabel("Az [m/s²]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_xlim(0.5, float(t[-1]))
    _save(fig, base_path)


def _write_report(
    path: Path,
    phase74: Path,
    phase75: Path,
    rows74: list[dict[str, str]],
    rows75: list[dict[str, str]],
) -> None:
    s4 = next(row for row in rows74 if row["scenario"] == "S4_classC_cornering")
    s8 = next(row for row in rows74 if row["scenario"] == "S8_worst_case")
    mismatch = next(row for row in rows75 if row["scenario"] == "mismatch_actual_0p6_assumed_0p8")
    matched = next(row for row in rows75 if row["scenario"] == "matched_mu_0p8")
    lines = [
        "# Phase 7.6 CarSim Final Figure Pack",
        "",
        f"- Phase 7.4 source: `{phase74}`",
        f"- Phase 7.5 source: `{phase75}`",
        "",
        "## Key Results",
        "",
        f"- S4 class-C cornering is now the mild comparison case: peak rho `{_f(s4, 'comfort_peak_rho'):.3f} -> {_f(s4, 'mpc_peak_rho'):.3f}`.",
        f"- S8 worst case is the headline case: peak rho `{_f(s8, 'comfort_peak_rho'):.3f} -> {_f(s8, 'mpc_peak_rho'):.3f}`, reduction `{100.0 * _f(s8, 'rho_reduction_ratio'):.1f}%`.",
        f"- S8 P95 rho improves `{_f(s8, 'comfort_rho_p95'):.3f} -> {_f(s8, 'mpc_rho_p95'):.3f}`.",
        f"- S8 minimum normal load improves `{_f(s8, 'comfort_min_fz_n'):.1f} N -> {_f(s8, 'mpc_min_fz_n'):.1f} N`.",
        f"- μ matched 0.8/0.8: peak rho `{_f(matched, 'comfort_peak_rho'):.3f} -> {_f(matched, 'mpc_peak_rho'):.3f}`.",
        f"- μ mismatch 0.6/0.8: peak rho `{_f(mismatch, 'comfort_peak_rho'):.3f} -> {_f(mismatch, 'mpc_peak_rho'):.3f}`, reduction `{100.0 * _f(mismatch, 'rho_reduction_ratio'):.1f}%`.",
        "",
        "## Generated Figures",
        "",
        "- `fig01_carsim_key_metrics.png/.pdf`: S4/S8 peak and P95 rho bars.",
        "- `fig02_carsim_s4_s8_rho_timeseries.png/.pdf`: S4 and S8 rho time histories.",
        "- `fig03_carsim_s8_load_force.png/.pdf`: S8 normal loads and MPC forces.",
        "- `fig04_carsim_mu_mismatch.png/.pdf`: matched and mismatch robustness.",
        "- `fig05_carsim_s8_speed_accel.png/.pdf`: S8 speed and body acceleration checks.",
        "",
        "## Data Files",
        "",
        "- `tables/phase7_key_metrics.csv`: compact table for paper/report.",
        "- `tables/phase7_4_summary.csv`: raw Stage 7.4 summary.",
        "- `tables/phase7_5_robustness_summary.csv`: raw Stage 7.5 summary.",
        "- `data/phase7_4_raw.npz`, `data/phase7_5_raw.npz`: full traces.",
        "",
        "Pacejka tire switching is intentionally omitted because the current FMU exposes μ inputs, not a tire-model selector.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(run_phase_7_6_carsim_final_figures())


if __name__ == "__main__":
    main()
