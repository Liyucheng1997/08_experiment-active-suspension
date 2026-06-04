"""Build Section V.F (CarSim) paper figures from Phase 7.4 raw data.

Paper-style (matching fig01-fig09) figures:
  fig10_carsim_s8.png       — 2-panel time series on S8 (top: ρ_max for
                              Comfort-QP vs Risk-MPC with ρ_safe annotated;
                              bottom: per-corner tire normal load showing
                              MPC re-loads the weakest wheel).
  fig11_carsim_s4_s8_summary.png — S4/S8 summary bars for peak ρ, P95 ρ,
                              and minimum normal load.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(r"F:/我的科研工作/05_Risk_Aware_Active_Suspension")
PAPER_FIG_DIR = Path(r"F:/latex/12_RiskAware_ActiveSuspension/figures")

PHASE = PROJECT / "results/paper_carsim_4000N_rate8000_fair_analytical_weights/phase-7.4_carsim_risk-mpc-validation_20260604_153240"
RAW_74 = PHASE / "raw.npz"

COLORS = {
    "comfort":   "#C83737",
    "mpc":       "#1F77B4",
    "safe":      "#2B2B2B",
    "FL":        "#0072B2",
    "FR":        "#E69F00",
    "RL":        "#009E73",
    "RR":        "#D55E00",
    "matched":   "#1F77B4",
    "mismatch":  "#E69F00",
}


def paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 450, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.8,
        "legend.fontsize": 8.6, "xtick.labelsize": 9.2, "ytick.labelsize": 9.2,
        "axes.grid": True, "grid.color": "#D0D0D0", "grid.alpha": 0.45,
        "grid.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round", "patch.linewidth": 0.0,
    })


# --------------------------------------------------------------------------
# Figure 10 — CarSim S8 main story
# --------------------------------------------------------------------------
def make_fig10_carsim_s8(out: Path) -> None:
    r = np.load(RAW_74, allow_pickle=True)
    sn = "S8_worst_case"

    t = r[f"{sn}_risk_mpc_time"]
    rho_c = r[f"{sn}_comfort_qp_rho_contact"].max(axis=1)
    rho_m = r[f"{sn}_risk_mpc_rho_contact"].max(axis=1)
    fz_c  = r[f"{sn}_comfort_qp_fz_true"]   # (N, 4) per-corner
    fz_m  = r[f"{sn}_risk_mpc_fz_true"]
    rho_safe = 0.85

    # The interesting window
    win = (t >= 0.5) & (t <= 4.0)

    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    ax0 = axes[0]
    ax0.plot(t[win], rho_c[win], color=COLORS["comfort"], label="Comfort-QP",
             linewidth=1.4)
    ax0.plot(t[win], rho_m[win], color=COLORS["mpc"],     label="Risk-MPC",
             linewidth=1.8)
    ax0.axhline(rho_safe, color=COLORS["safe"], linestyle=(0, (4, 2)),
                linewidth=1.0, label=r"$\rho_{safe}=0.85$")
    peak_c = float(np.max(rho_c[win])); peak_m = float(np.max(rho_m[win]))
    ax0.annotate(f"peak={peak_c:.2f}", xy=(t[win][int(np.argmax(rho_c[win]))], peak_c),
                 textcoords="offset points", xytext=(5, 5),
                 fontsize=8.4, color=COLORS["comfort"])
    ax0.annotate(f"peak={peak_m:.2f}", xy=(t[win][int(np.argmax(rho_m[win]))], peak_m),
                 textcoords="offset points", xytext=(5, -14),
                 fontsize=8.4, color=COLORS["mpc"])
    ax0.set_ylabel(r"$\rho_{max}$")
    ax0.set_title(
        f"S8 on CarSim Pacejka plant: Risk-MPC cuts peak $\\rho$ "
        f"by {(peak_c - peak_m)/peak_c*100:.1f}%"
    )
    ax0.legend(loc="upper right", ncol=3, fontsize=8.4)

    ax1 = axes[1]
    # Plot the WEAKEST wheel load — typically the wheel that minimizes F_z under
    # combined brake+cornering load transfer. Pick the corner with the lowest
    # comfort F_z within the window.
    weakest = int(np.argmin(fz_c[win].min(axis=0)))
    corner_names = ["FL", "FR", "RL", "RR"]
    cname = corner_names[weakest]
    ax1.plot(t[win], fz_c[win, weakest], color=COLORS["comfort"], linewidth=1.4,
             label=f"Comfort-QP $F_{{z,{cname}}}$")
    ax1.plot(t[win], fz_m[win, weakest], color=COLORS["mpc"], linewidth=1.8,
             label=f"Risk-MPC $F_{{z,{cname}}}$")
    ax1.set_ylabel("Tire normal load [N]")
    ax1.set_xlabel("Time [s]")
    ax1.set_title(f"Weakest corner ({cname}): Risk-MPC raises the floor of $F_z$")
    ax1.legend(loc="lower right", fontsize=8.4)
    ax1.set_xlim(t[win].min(), t[win].max())

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 11 — S4/S8 CarSim summary
# --------------------------------------------------------------------------
def make_fig11_carsim_s4_s8_summary(out: Path) -> None:
    r = np.load(RAW_74, allow_pickle=True)
    scenarios = ["S4_classC_cornering", "S8_worst_case"]
    labels = ["S4", "S8"]
    rho_safe = 0.85

    peak_c, peak_m = [], []
    p95_c, p95_m = [], []
    minfz_c, minfz_m = [], []
    for scenario in scenarios:
        t = r[f"{scenario}_comfort_qp_time"]
        vx = r[f"{scenario}_comfort_qp_vx_kmh"]
        fz_c = r[f"{scenario}_comfort_qp_fz_true"]
        fz_m = r[f"{scenario}_risk_mpc_fz_true"]
        rho_c = r[f"{scenario}_comfort_qp_rho_contact"].max(axis=1)
        rho_m = r[f"{scenario}_risk_mpc_rho_contact"].max(axis=1)
        valid = (t >= 0.5) & (vx > 1.0) & np.all(fz_c > 100.0, axis=1) & np.all(fz_m > 100.0, axis=1)
        peak_c.append(float(np.max(rho_c[valid])))
        peak_m.append(float(np.max(rho_m[valid])))
        p95_c.append(float(np.percentile(rho_c[valid], 95)))
        p95_m.append(float(np.percentile(rho_m[valid], 95)))
        minfz_c.append(float(np.min(fz_c[valid])))
        minfz_m.append(float(np.min(fz_m[valid])))

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.1))
    x = np.arange(len(labels))
    width = 0.34

    panels = [
        (axes[0], peak_c, peak_m, r"Peak $\rho_{max}$", "Peak utilization"),
        (axes[1], p95_c, p95_m, r"P95 $\rho_{max}$", "Sustained utilization"),
        (axes[2], minfz_c, minfz_m, "min $F_z$ [N]", "Normal-load floor"),
    ]
    for ax, comfort, mpc, ylabel, title in panels:
        ax.bar(x - width / 2, comfort, width, color=COLORS["comfort"], alpha=0.92, label="Comfort-QP")
        ax.bar(x + width / 2, mpc, width, color=COLORS["mpc"], alpha=0.95, label="Risk-MPC")
        if "rho" in ylabel:
            ax.axhline(rho_safe, color=COLORS["safe"], linestyle=(0, (4, 2)), linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(loc="upper right", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    paper_style()
    out_dir = PROJECT / "results/paper_fig_polish"
    out_dir.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig10 = out_dir / "fig10_carsim_s8.png"
    fig11 = out_dir / "fig11_carsim_s4_s8_summary.png"
    make_fig10_carsim_s8(fig10)
    make_fig11_carsim_s4_s8_summary(fig11)

    for fig in (fig10, fig11):
        dst = PAPER_FIG_DIR / fig.name
        shutil.copy2(fig, dst)
        print(f"copied {fig.name} -> {dst}")


if __name__ == "__main__":
    main()
