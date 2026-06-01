from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import RESIDUAL_DECAY, FullCarRiskMPC
from risk_aware_active_suspension.controllers.risk_qp import RiskQPParams
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.risk_mpc_ablation import ABLATIONS, _run_ablation_closed_loop, _ablation_risk_params
from risk_aware_active_suspension.scenarios.risk_mpc_residual_prediction import run_mpc_residual_closed_loop
from risk_aware_active_suspension.scenarios.risk_mpc_scenario_sweep import (
    PHASE6_SCENARIOS,
    _build_inputs,
    _risk_params_for,
    _run_comfort,
)
from risk_aware_active_suspension.utils.config import from_yaml, lqr_from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


PHASE6_SOURCE = Path("results/phase-6_risk-mpc_scenario-sweep-main_20260528_183607")
ABLATION_SOURCE = Path("results/phase-6.6_risk-mpc_component-ablation_20260528_185010")

COLORS = {
    "comfort": "#C83737",
    "mpc": "#1F77B4",
    "qp": "#6F6F6F",
    "safe": "#2B2B2B",
    "sto": "#E69F00",
    "sigma": "#D55E00",
    "tight": "#7A3FB0",
}


def run_phase_6_8_paper_figures(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    observer_config: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
    phase6_source: str | Path = PHASE6_SOURCE,
    ablation_source: str | Path = ABLATION_SOURCE,
    dt: float = 0.001,
) -> Path:
    _set_paper_style()
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    observer_config = Path(observer_config)
    phase6_source = Path(phase6_source)
    ablation_source = Path(ablation_source)
    vehicle = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)
    observer_params = observer_from_yaml(observer_config)
    plant = FullCar(vehicle)

    logger = RunLogger.create(results_root, phase="phase-6.8", step="paper", descriptor="figure-pack")
    fig_dir = logger.run_dir / "figures"
    table_dir = logger.run_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    for src, name in [
        (phase6_source / "phase6_summary.csv", "phase6_summary.csv"),
        (ablation_source / "phase6_6_ablation_summary.csv", "phase6_6_ablation_summary.csv"),
    ]:
        shutil.copy2(src, table_dir / name)

    phase6_rows = _read_rows(table_dir / "phase6_summary.csv")
    ablation_rows = _read_rows(table_dir / "phase6_6_ablation_summary.csv")
    _plot_main_peak_rho(fig_dir / "fig01_scenario_peak_rho.png", phase6_rows)
    _plot_safety_comfort_tradeoff(fig_dir / "fig02_safety_comfort_tradeoff.png", phase6_rows)

    runs = _rerun_selected_full_cases(plant, vehicle, lqr, observer_params, dt)
    _plot_s8_timeseries(fig_dir / "fig03_s8_worst_case_timeseries.png", runs["S8_worst_case"])
    _plot_s9_timeseries(fig_dir / "fig04_s9_corner_bump_timeseries.png", runs["S9_corner_bump"])

    _plot_ablation_peak_p95(fig_dir / "fig05_ablation_safety_metrics.png", ablation_rows)
    _plot_ablation_tradeoff(fig_dir / "fig06_ablation_tradeoff.png", ablation_rows)

    ablation_runs = _rerun_s9_ablation_cases(plant, vehicle, lqr, observer_params, dt)
    _plot_s9_ablation_timeseries(fig_dir / "fig07_s9_ablation_timeseries.png", ablation_runs)

    _write_analysis_doc(logger.run_dir / "paper_experiment_analysis.md", phase6_rows, ablation_rows)
    logger.log("Phase 6.8 paper figure pack completed.")
    return logger.run_dir


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.8,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.0,
        }
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _plot_main_peak_rho(path: Path, rows: list[dict[str, str]]) -> None:
    names = [r["scenario"].split("_", 1)[0] for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    comfort = [_f(r, "comfort_peak_rho") for r in rows]
    mpc = [_f(r, "mpc_peak_rho") for r in rows]
    high_risk_start_idx = 6  # S7, S8, S9 are high-risk per fig02 logic

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    # Shaded bands for the low-risk and high-risk regions
    ax.axvspan(-0.5, high_risk_start_idx - 0.5, color="#F0F0F0", alpha=0.7, zorder=0)
    ax.axvspan(high_risk_start_idx - 0.5, len(rows) - 0.5, color="#FBEAD9", alpha=0.55, zorder=0)
    ax.bar(x - width / 2, comfort, width, label="Comfort-QP", color=COLORS["comfort"], alpha=0.92, zorder=2)
    ax.bar(x + width / 2, mpc, width, label="Risk-MPC", color=COLORS["mpc"], alpha=0.95, zorder=2)
    ax.axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.1, label=r"$\rho_{safe}=0.85$", zorder=3)

    for idx, row in enumerate(rows):
        reduction = 100.0 * _f(row, "rho_reduction_ratio")
        if abs(reduction) >= 10.0:
            ax.text(idx, max(comfort[idx], mpc[idx]) + 0.06, f"{reduction:.0f}%",
                    ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#0B4D80")

    # Region headers above the bars
    top_y = 2.45
    ax.text((high_risk_start_idx - 1) / 2.0, top_y, "Low risk (S1–S6)",
            ha="center", va="top", fontsize=10.0, fontweight="bold",
            color="#555555")
    ax.text((high_risk_start_idx + len(rows) - 1) / 2.0, top_y,
            "High risk (S7–S9)",
            ha="center", va="top", fontsize=10.0, fontweight="bold",
            color="#8C4500")
    # Faint vertical separator between low- and high-risk regions
    ax.axvline(high_risk_start_idx - 0.5, color="#999999",
               linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"Peak tire utilization $\rho_{max}$")
    ax.set_title("Scenario sweep: risk-MPC engages only when the friction margin is threatened")
    ax.set_ylim(0.0, 2.55)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.legend(ncol=3, loc="center left", frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_safety_comfort_tradeoff(path: Path, rows: list[dict[str, str]]) -> None:
    labels = [r["scenario"].split("_", 1)[0] for r in rows]
    reduction = [100.0 * _f(r, "rho_reduction_ratio") for r in rows]
    heave_delta = [_f(r, "mpc_heave_rms") - _f(r, "comfort_heave_rms") for r in rows]
    high_risk_labels = {"S7", "S8", "S9"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # Lightly shade the "useful work" quadrant: positive reduction, any heave change.
    xlim_pad_left = min(heave_delta) - 0.012
    xlim_pad_right = max(heave_delta) + 0.025
    ylim_pad_bot = min(reduction) - 8.0
    ylim_pad_top = max(reduction) + 10.0
    ax.axhspan(0.0, ylim_pad_top, color="#E8F1FA", alpha=0.45, zorder=0)
    ax.text(
        xlim_pad_right - 0.001, ylim_pad_top - 1.0,
        "Safety gain at modest comfort cost",
        ha="right", va="top", fontsize=8.4, color="#0B4D80", style="italic",
    )
    ax.text(
        xlim_pad_left + 0.001, ylim_pad_bot + 2.0,
        "Comfort-QP wins (low-risk)",
        ha="left", va="bottom", fontsize=8.4, color="#555555", style="italic",
    )

    ax.axhline(0, color="#777777", linewidth=0.9, zorder=1)
    ax.axvline(0, color="#777777", linewidth=0.9, zorder=1)
    for x_i, y_i, label in zip(heave_delta, reduction, labels):
        is_high = label in high_risk_labels
        ax.scatter(
            [x_i], [y_i], s=80 if is_high else 52,
            color=COLORS["mpc"] if is_high else "#9C9C9C",
            edgecolor="white", linewidth=0.8, zorder=3,
        )

    offsets = {
        "S1": (6, 6),
        "S2": (-22, -2),
        "S3": (-22, 8),
        "S4": (6, 6),
        "S5": (6, -12),
        "S6": (6, -10),
        "S7": (8, 4),
        "S8": (8, 4),
        "S9": (8, 4),
    }
    for x_i, y_i, label in zip(heave_delta, reduction, labels):
        ax.annotate(label, (x_i, y_i),
                    textcoords="offset points",
                    xytext=offsets.get(label, (5, 4)),
                    fontsize=9.2,
                    fontweight=("bold" if label in high_risk_labels else "normal"))

    ax.set_xlabel(r"Heave RMS change vs Comfort-QP [m/s$^2$]")
    ax.set_ylabel(r"Peak $\rho_{max}$ reduction [%]")
    ax.set_title("Safety-comfort trade-off: gain concentrated in high-risk scenarios")
    ax.set_xlim(xlim_pad_left, xlim_pad_right)
    ax.set_ylim(ylim_pad_bot, ylim_pad_top)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _rerun_selected_full_cases(plant, vehicle, lqr, observer_params, dt: float) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for scenario_name in ("S8_worst_case", "S9_corner_bump"):
        spec = next(s for s in PHASE6_SCENARIOS if s.name == scenario_name)
        t, roads, a_x, a_y, f_c = _build_inputs(plant, spec, dt)
        risk = _risk_params_for(spec)
        qp = RiskQPParams(rho_safe=0.85, enforce_rate=False, r_du_factor=1.0e-5, osqp_eps_abs=1.0e-5, osqp_eps_rel=1.0e-5, osqp_max_iter=100000)
        comfort = _run_comfort(FullCarComfortQP(vehicle, lqr), plant, t, roads, a_x, a_y, f_c, spec.mu)
        mpc = run_mpc_residual_closed_loop(
            FullCarRiskMPC(vehicle, lqr, risk, qp, horizon=10, warm_start=False, reuse_solver=True),
            plant,
            t,
            roads,
            a_y,
            f_c,
            spec.mu,
            observer_params,
            prediction_mode=RESIDUAL_DECAY,
            alpha=0.95,
            fz_tightening_n=observer_params.delta_fz_err,
            a_x=a_x,
        )
        out[scenario_name] = {"t": t, "comfort": comfort, "mpc": mpc}
    return out


def _plot_s8_timeseries(path: Path, run: dict[str, np.ndarray]) -> None:
    t = run["t"]
    comfort = run["comfort"]
    mpc = run["mpc"]
    rho_c = np.max(comfort["rho_contact"], axis=1)
    rho_m = np.max(mpc["rho_contact"], axis=1)
    # Smoothed peak envelope for the noisy ρ_max trace (50 ms window).
    win_samp = max(1, int(0.05 / (t[1] - t[0])))
    kernel = np.ones(win_samp) / win_samp
    pad = win_samp // 2
    rho_c_smooth = np.convolve(np.pad(rho_c, pad, mode="edge"), kernel, mode="valid")[: len(t)]
    rho_m_smooth = np.convolve(np.pad(rho_m, pad, mode="edge"), kernel, mode="valid")[: len(t)]

    peak_c_idx = int(np.argmax(rho_c))
    peak_m_idx = int(np.argmax(rho_m))

    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    ax0 = axes[0]
    ax0.plot(t, rho_c, color=COLORS["comfort"], alpha=0.32, linewidth=0.9)
    ax0.plot(t, rho_m, color=COLORS["mpc"], alpha=0.32, linewidth=0.9)
    ax0.plot(t, rho_c_smooth, label="Comfort-QP (50 ms env.)", color=COLORS["comfort"], linewidth=2.0)
    ax0.plot(t, rho_m_smooth, label="Risk-MPC (50 ms env.)", color=COLORS["mpc"], linewidth=2.0)
    ax0.axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.0, label=r"$\rho_{safe}$")
    ax0.scatter([t[peak_c_idx]], [rho_c[peak_c_idx]],
                marker="v", color=COLORS["comfort"], s=70, zorder=4, edgecolor="white", linewidth=0.7)
    ax0.scatter([t[peak_m_idx]], [rho_m[peak_m_idx]],
                marker="v", color=COLORS["mpc"], s=70, zorder=4, edgecolor="white", linewidth=0.7)
    ax0.annotate(f"peak={rho_c[peak_c_idx]:.2f}", (t[peak_c_idx], rho_c[peak_c_idx]),
                 textcoords="offset points", xytext=(6, 6), fontsize=8.4, color=COLORS["comfort"])
    ax0.annotate(f"peak={rho_m[peak_m_idx]:.2f}", (t[peak_m_idx], rho_m[peak_m_idx]),
                 textcoords="offset points", xytext=(6, -14), fontsize=8.4, color=COLORS["mpc"])
    ax0.set_ylabel(r"$\rho_{max}$")
    ax0.set_title("S8 worst-case (class-D + brake + cornering, μ=0.7): Risk-MPC cuts peak ρ 38%")
    ax0.legend(loc="upper right", ncol=2, fontsize=8.2)
    ax0.set_xlim(0.0, t[-1])

    ax1 = axes[1]
    ax1.plot(t, mpc["forces"][:, 0], color="#0072B2", label="FL")
    ax1.plot(t, mpc["forces"][:, 1], color="#E69F00", label="FR")
    ax1.plot(t, mpc["forces"][:, 2], color="#009E73", label="RL")
    ax1.plot(t, mpc["forces"][:, 3], color="#D55E00", label="RR")
    ax1.set_ylabel("Actuator force [N]")
    ax1.set_xlabel("Time [s]")
    ax1.legend(ncol=4, loc="upper right", fontsize=8.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_s9_timeseries(path: Path, run: dict[str, np.ndarray]) -> None:
    t = run["t"]
    comfort = run["comfort"]
    mpc = run["mpc"]
    rho_c = np.max(comfort["rho_contact"], axis=1)
    rho_m = np.max(mpc["rho_contact"], axis=1)
    # The bump fires at t = 1.8 in the scenario definition.
    bump_t = 1.8

    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    for ax in axes:
        ax.axvline(bump_t, color="#777777", linestyle=":", linewidth=0.9, zorder=1)

    ax0 = axes[0]
    ax0.plot(t, rho_c, label="Comfort-QP", color=COLORS["comfort"], linewidth=1.6)
    ax0.plot(t, rho_m, label="Risk-MPC", color=COLORS["mpc"], linewidth=1.6)
    ax0.axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)),
                linewidth=1.0, label=r"$\rho_{safe}$")
    peak_c_idx = int(np.argmax(rho_c))
    peak_m_idx = int(np.argmax(rho_m))
    ax0.scatter([t[peak_c_idx]], [rho_c[peak_c_idx]], marker="v", s=55,
                color=COLORS["comfort"], zorder=4, edgecolor="white", linewidth=0.6)
    ax0.scatter([t[peak_m_idx]], [rho_m[peak_m_idx]], marker="v", s=55,
                color=COLORS["mpc"], zorder=4, edgecolor="white", linewidth=0.6)
    # Peak labels placed BELOW the peak markers to avoid the title.
    ax0.annotate(f"peak={rho_c[peak_c_idx]:.2f}",
                 (t[peak_c_idx], rho_c[peak_c_idx]),
                 textcoords="offset points", xytext=(8, -14),
                 fontsize=8.6, color=COLORS["comfort"])
    ax0.annotate(f"peak={rho_m[peak_m_idx]:.2f}",
                 (t[peak_m_idx], rho_m[peak_m_idx]),
                 textcoords="offset points", xytext=(8, -14),
                 fontsize=8.6, color=COLORS["mpc"])
    ax0.text(bump_t + 0.02, 0.32, "bump", fontsize=8.6,
             color="#444444", style="italic")
    ax0.set_ylabel(r"$\rho_{max}$")
    ax0.set_title("S9: outer-front bump during cornering — STO opens the dynamic risk channel")
    ax0.set_ylim(0.0, max(rho_c.max(), rho_m.max()) * 1.20)
    ax0.legend(loc="upper right", ncol=3, fontsize=8.4)

    ax1 = axes[1]
    ax1.plot(t, mpc["forces"][:, 1], color="#E69F00", label="FR (bumped)", linewidth=1.8)
    ax1.plot(t, mpc["forces"][:, 0], color="#0072B2", label="FL", alpha=0.75)
    ax1.plot(t, mpc["forces"][:, 2], color="#009E73", label="RL", alpha=0.75)
    ax1.plot(t, mpc["forces"][:, 3], color="#D55E00", label="RR", alpha=0.75)
    ax1.set_ylabel("Actuator force [N]")
    ax1.set_xlabel("Time [s]")
    ax1.legend(ncol=4, loc="lower right", fontsize=8.4)
    axes[0].set_xlim(0.75, 3.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_ablation_peak_p95(path: Path, rows: list[dict[str, str]]) -> None:
    # A3 (κ_ρ = 0) collapses to Full exactly in this scenario set, so we omit
    # it from the bar chart and call out the result in the caption.
    variants = ["full", "A1_no_sto", "A2_sigma_zero", "A4_no_tightening"]
    labels = ["Full", "No STO", r"$\sigma_\rho=0$", "No tightening"]
    bar_colors_peak = [COLORS["mpc"], COLORS["sto"], COLORS["sigma"], COLORS["tight"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=False)
    for ax, scenario in zip(axes, ["S8_worst_case", "S9_corner_bump"]):
        peak = [_metric(rows, scenario, variant, "peak_rho") for variant in variants]
        p95 = [_metric(rows, scenario, variant, "p95_rho") for variant in variants]
        full_peak = peak[0]
        full_p95 = p95[0]
        x = np.arange(len(variants))
        bars_peak = ax.bar(x - 0.20, peak, 0.36, color=bar_colors_peak, alpha=0.95, label="Peak")
        bars_p95 = ax.bar(x + 0.20, p95, 0.36, color=[_lighten(c, 0.55) for c in bar_colors_peak],
                          alpha=0.95, label="P95")
        ax.axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.0, label=r"$\rho_{safe}$")
        ax.axhline(full_peak, color=COLORS["mpc"], linestyle=":", linewidth=0.8, alpha=0.6)
        # Annotate delta vs full above each ablation bar
        for i, (pk, p9) in enumerate(zip(peak, p95)):
            if i == 0:
                continue
            d_peak = pk - full_peak
            ax.text(x[i] - 0.20, pk + 0.04, f"{d_peak:+.2f}",
                    ha="center", va="bottom", fontsize=8.0, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(scenario.split("_", 1)[0])
        ax.set_ylabel(r"Tire utilization $\rho$")
        ax.set_ylim(0.0, max(max(peak), max(p95)) * 1.18)
    axes[0].legend(loc="upper left", fontsize=8.4)
    fig.suptitle("Ablation: each contribution removed (κ_ρ=0 identical to Full, omitted)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _lighten(hex_color: str, factor: float) -> str:
    """Return a lighter shade of ``hex_color`` blended with white by ``factor``."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r2 = int(round(r + (255 - r) * factor))
    g2 = int(round(g + (255 - g) * factor))
    b2 = int(round(b + (255 - b) * factor))
    return f"#{r2:02X}{g2:02X}{b2:02X}"


def _plot_ablation_tradeoff(path: Path, rows: list[dict[str, str]]) -> None:
    variants = ["A1_no_sto", "A2_sigma_zero", "A4_no_tightening"]
    pretty = {"A1_no_sto": "No STO", "A2_sigma_zero": r"$\sigma_\rho=0$", "A4_no_tightening": "No tightening"}
    palette = {"A1_no_sto": COLORS["sto"], "A2_sigma_zero": COLORS["sigma"], "A4_no_tightening": COLORS["tight"]}
    scenarios = [("S8_worst_case", "S8"), ("S9_corner_bump", "S9")]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    for ax, (scenario, title) in zip(axes, scenarios):
        ax.axhline(0, color="#777777", linewidth=0.9)
        ax.axvline(0, color="#777777", linewidth=0.9)
        for variant in variants:
            row = next(r for r in rows if r["scenario"] == scenario and r["ablation"] == variant)
            x = _f(row, "heave_delta_vs_full")
            y = _f(row, "p95_rho_delta_vs_full")
            ax.scatter([x], [y], s=110, marker="o", color=palette[variant],
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(pretty[variant], (x, y),
                        textcoords="offset points", xytext=(8, 6), fontsize=8.8,
                        color=palette[variant], fontweight="bold")
        ax.set_xlabel(r"Heave RMS change vs full [m/s$^2$]")
        ax.set_title(title)
    axes[0].set_ylabel(r"P95 $\rho_{max}$ change vs full")
    fig.suptitle("Ablation safety / comfort trade-off (positive ⇒ degraded margin)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _metric(rows: list[dict[str, str]], scenario: str, ablation: str, metric: str) -> float:
    return _f(next(r for r in rows if r["scenario"] == scenario and r["ablation"] == ablation), metric)


def _rerun_s9_ablation_cases(plant, vehicle, lqr, observer_params, dt: float) -> dict[str, dict[str, np.ndarray]]:
    spec = next(s for s in PHASE6_SCENARIOS if s.name == "S9_corner_bump")
    t, roads, a_x, a_y, f_c = _build_inputs(plant, spec, dt)
    out: dict[str, dict[str, np.ndarray]] = {"t": {"value": t}}  # type: ignore[dict-item]
    qp = RiskQPParams(rho_safe=0.85, enforce_rate=False, r_du_factor=1.0e-5, osqp_eps_abs=1.0e-5, osqp_eps_rel=1.0e-5, osqp_max_iter=100000)
    base = _risk_params_for(spec)
    for ablation in ABLATIONS:
        run = _run_ablation_closed_loop(
            FullCarRiskMPC(vehicle, lqr, _ablation_risk_params(base, ablation), qp, horizon=10, warm_start=False, reuse_solver=True),
            plant,
            t,
            roads,
            a_x,
            a_y,
            f_c,
            spec.mu,
            observer_params,
            disable_sto=ablation.disable_sto,
            fz_tightening_n=0.0 if ablation.disable_tightening else observer_params.delta_fz_err,
        )
        out[ablation.name] = run
    return out


def _plot_s9_ablation_timeseries(path: Path, runs: dict[str, dict[str, np.ndarray]]) -> None:
    t = runs["t"]["value"]  # type: ignore[index]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    styles = {
        "full": ("Full risk-MPC", COLORS["mpc"], 2.2, 1.0),
        "A1_no_sto": ("No STO", COLORS["sto"], 1.8, 0.95),
        "A2_sigma_zero": (r"$\sigma_\rho=0$ (risk off)", COLORS["sigma"], 1.8, 0.95),
        "A4_no_tightening": ("No tightening", COLORS["tight"], 1.6, 0.85),
    }
    for key, (label, color, lw, alpha) in styles.items():
        ax.plot(
            t, np.max(runs[key]["rho_contact"], axis=1),
            label=label, color=color, linewidth=lw, alpha=alpha,
        )
    ax.axhline(0.85, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.1,
               label=r"$\rho_{safe}=0.85$")
    ax.axvline(1.8, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(1.802, 1.92, "bump impact", fontsize=8.6, color="#444444", style="italic")
    ax.set_xlim(1.68, 2.55)
    ax.set_ylim(0.45, 2.15)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\rho_{max}$")
    ax.set_title("S9 ablation: removing the STO or σ_ρ exposes ρ to the bump (purple ≈ blue)")
    ax.legend(loc="upper right", fontsize=8.4, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_analysis_doc(path: Path, phase6_rows: list[dict[str, str]], ablation_rows: list[dict[str, str]]) -> None:
    s7 = next(r for r in phase6_rows if r["scenario"] == "S7_brake_cornering")
    s8 = next(r for r in phase6_rows if r["scenario"] == "S8_worst_case")
    s9 = next(r for r in phase6_rows if r["scenario"] == "S9_corner_bump")
    low = [r for r in phase6_rows if r["scenario"].startswith(("S1", "S2", "S3"))]
    low_max_change = max(abs(100.0 * _f(r, "rho_reduction_ratio")) for r in low)
    lines = [
        "# Paper Experiment Analysis: Risk-Aware Active Suspension",
        "",
        "## Story",
        "",
        "本文实验线索建议按三层展开：",
        "",
        "1. 在低风险工况中，risk-MPC 不应无意义介入，证明风险调度器不会牺牲正常舒适性。",
        "2. 在制动、转弯、粗糙路和瞬态 bump 的高风险工况中，risk-MPC 通过预测 horizon 主动调配垂向力，降低轮胎利用率峰值。",
        "3. Ablation 证明效果来自三个实际通道：STO/路面残差、风险激活 sigma、以及 Fz 误差 tightening；同时当前 kappa 局部权重通道贡献不明显，应作为诊断结论如实说明。",
        "",
        "## Main Scenario Sweep",
        "",
        f"- Low-risk scenarios S1-S3: 最大 peak rho 变化约 {low_max_change:.1f}%，说明控制器在非临界工况下基本保持 comfort-QP 行为。",
        f"- S7 braking + cornering: peak rho 从 {_f(s7, 'comfort_peak_rho'):.3f} 降到 {_f(s7, 'mpc_peak_rho'):.3f}，降低 {100*_f(s7, 'rho_reduction_ratio'):.1f}%。",
        f"- S8 scaled class-D + braking + cornering: peak rho 从 {_f(s8, 'comfort_peak_rho'):.3f} 降到 {_f(s8, 'mpc_peak_rho'):.3f}，降低 {100*_f(s8, 'rho_reduction_ratio'):.1f}%。这是主 paper scenario，体现 horizon prediction 相比 comfort-QP 的安全裕度收益。",
        f"- S9 cornering + outer-front bump: peak rho 从 {_f(s9, 'comfort_peak_rho'):.3f} 降到 {_f(s9, 'mpc_peak_rho'):.3f}，降低 {100*_f(s9, 'rho_reduction_ratio'):.1f}%。这是 STO/动态路面风险通道的核心展示。",
        f"- S8/S9 的 heave RMS 分别增加 {100*_f(s8, 'heave_degradation_ratio'):.1f}% 和 {100*_f(s9, 'heave_degradation_ratio'):.1f}%。这应解释为安全-舒适 trade-off，而不是 comfort objective 失效。",
        "",
        "## Ablation Interpretation",
        "",
    ]
    for scenario in ("S8_worst_case", "S9_corner_bump"):
        full = next(r for r in ablation_rows if r["scenario"] == scenario and r["ablation"] == "full")
        a1 = next(r for r in ablation_rows if r["scenario"] == scenario and r["ablation"] == "A1_no_sto")
        a2 = next(r for r in ablation_rows if r["scenario"] == scenario and r["ablation"] == "A2_sigma_zero")
        a4 = next(r for r in ablation_rows if r["scenario"] == scenario and r["ablation"] == "A4_no_tightening")
        lines.extend(
            [
                f"### {scenario}",
                "",
                f"- Full risk-MPC: peak rho={_f(full, 'peak_rho'):.3f}, P95 rho={_f(full, 'p95_rho'):.3f}, heave RMS={_f(full, 'heave_rms'):.3f}.",
                f"- A1 no-STO: peak delta={_f(a1, 'peak_rho_delta_vs_full'):+.3f}, P95 delta={_f(a1, 'p95_rho_delta_vs_full'):+.3f}, heave delta={_f(a1, 'heave_delta_vs_full'):+.3f}.",
                f"- A2 sigma=0: peak delta={_f(a2, 'peak_rho_delta_vs_full'):+.3f}, P95 delta={_f(a2, 'p95_rho_delta_vs_full'):+.3f}. This is the cleanest evidence that risk activation is necessary.",
                f"- No tightening: peak delta={_f(a4, 'peak_rho_delta_vs_full'):+.3f}, P95 delta={_f(a4, 'p95_rho_delta_vs_full'):+.3f}. Tightening mainly protects against observer/model residual error.",
                "",
            ]
        )
    lines.extend(
        [
            "## C2 Limitation: κ_ρ Local Weight Has No Observable Effect",
            "",
            "Ablation A3 (κ_ρ = 0) coincides exactly with the Full risk-MPC on both S8 and S9: all",
            "deltas (peak ρ, P95 ρ, heave RMS, actuator effort) are 0.000. Two interpretations are",
            "consistent with the data:",
            "",
            "1. *Mechanism explanation* — once the global σ_ρ activates the protection cost, the",
            "   MPC's horizon prediction already directs effort to the wheel whose F_z is dropping,",
            "   so the per-wheel q_p amplification has no incremental optimization leverage in our",
            "   linear-plant + slack-with-quadratic-penalty formulation.",
            "2. *Scenario explanation* — in S1-S9 the wheels enter the risk band roughly together",
            "   (rough-road cornering or symmetric bump), so the per-wheel ρ spread is small and",
            "   max(0, ρ_ij - ρ_th) is similar across wheels.",
            "",
            "Recommended paper edits:",
            "- Drop the `local weight focuses effort on the critical wheel` sub-claim from C2.",
            "- Keep the sigmoid scheduler + slack reformulation + STO observer-error tightening",
            "  as the surviving contributions.",
            "- Either add a split-μ scenario (asymmetric ρ spread) to revive κ_ρ as a contribution,",
            "  or report the no-effect outcome as honest negative evidence in the limitations.",
            "",
            "## Figure Usage",
            "",
            "- `fig01_scenario_peak_rho.png`: main scenario sweep, low-risk and high-risk regimes shaded.",
            "- `fig02_safety_comfort_tradeoff.png`: safety-comfort trade-off; high-risk scenarios highlighted.",
            "- `fig03_s8_worst_case_timeseries.png`: S8 headline; 50 ms envelope + peak markers.",
            "- `fig04_s9_corner_bump_timeseries.png`: STO/transient bump story with bump line at t=1.8 s.",
            "- `fig05_ablation_safety_metrics.png`: ablation bar chart, A3 omitted (no effect; noted in suptitle).",
            "- `fig06_ablation_tradeoff.png`: ablation trade-off, faceted by scenario.",
            "- `fig07_s9_ablation_timeseries.png`: transient ablation comparison with bump line.",
            "",
            "## Notes For Writing",
            "",
            "- 不建议宣称所有高风险工况都严格压到 rho_safe 以下；当前线性模型和执行器约束下，更准确的说法是 risk-MPC substantially reduces peak utilization and improves margin.",
            "- S9 的 no-STO ablation 在 peak rho 上低于 full，但代价是 heave RMS 和 actuator force 明显变大，因此应从 safety-comfort-actuation trade-off 解释，而不是只看单个 peak 指标。",
            "- A3 kappa_rho=0 没有可观测影响，论文 Conclusion 必须重写：本工作的 Conclusion 草稿（PDF 末页）写的是别的工作（RFG），必须替换为对齐 abstract 的版本，引用本文实测数 S7 21%/S8 38%/S9 31% 和 A1/A2/A4 ablation 结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(run_phase_6_8_paper_figures())


if __name__ == "__main__":
    main()
