"""Build Section V.F (CarSim) paper figures from Phase 7.6 raw data.

Paper-style (matching fig01-fig09) figures:
  fig10_carsim_s8.png       — 2-panel time series on S8 (top: ρ_max for
                              Comfort-QP vs Risk-MPC with ρ_safe annotated;
                              bottom: per-corner tire normal load showing
                              MPC re-loads the weakest wheel).
  fig11_carsim_mu_mismatch.png — μ-mismatch robustness (left: bar chart of
                              peak ρ for matched vs mismatched μ; right:
                              time series of Risk-MPC ρ_max under both).
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(r"F:/我的科研工作/05_Risk_Aware_Active_Suspension")
PAPER_FIG_DIR = Path(
    r"D:/OneDrive - Unimore/02_博士相关资料/05_论文资料备份/"
    r"01_我的Latex论文写作/12_Risk-Aware Active Suspension Control "
    r"for Tire Friction Margin Protection Using Super-Twisting Normal "
    r"Load Estimation/figures"
)

PHASE = PROJECT / "results/phase-7.6_carsim_final-figure-pack_20260528_224921"
RAW_74 = PHASE / "data/phase7_4_raw.npz"
RAW_75 = PHASE / "data/phase7_5_raw.npz"

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
# Figure 11 — μ mismatch robustness
# --------------------------------------------------------------------------
def make_fig11_carsim_mu(out: Path) -> None:
    r5 = np.load(RAW_75, allow_pickle=True)

    # The Phase 7.5 raw npz stores two cases under the keys "matched_mu_0p8"
    # and "mismatch_actual_0p6_assumed_0p8".  Time-series we need:
    case_keys = {
        "matched":  "matched_mu_0p8",
        "mismatch": "mismatch_actual_0p6_assumed_0p8",
    }
    t   = r5[f"{case_keys['matched']}_risk_mpc_time"]
    rho_matched_c = r5[f"{case_keys['matched']}_comfort_qp_rho_contact"].max(axis=1)
    rho_matched_m = r5[f"{case_keys['matched']}_risk_mpc_rho_contact"].max(axis=1)
    rho_mis_c     = r5[f"{case_keys['mismatch']}_comfort_qp_rho_contact"].max(axis=1)
    rho_mis_m     = r5[f"{case_keys['mismatch']}_risk_mpc_rho_contact"].max(axis=1)

    # Summary numbers (peak ρ for the bar chart)
    peak_matched_c  = float(np.max(rho_matched_c))
    peak_matched_m  = float(np.max(rho_matched_m))
    peak_mis_c      = float(np.max(rho_mis_c))
    peak_mis_m      = float(np.max(rho_mis_m))
    rho_safe = 0.85

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.5))

    # ---- Left: paired bar chart ----
    ax0 = axes[0]
    x = np.arange(2)
    width = 0.34
    bars_c = ax0.bar(x - width/2, [peak_matched_c, peak_mis_c], width,
                     color=COLORS["comfort"], alpha=0.92, label="Comfort-QP")
    bars_m = ax0.bar(x + width/2, [peak_matched_m, peak_mis_m], width,
                     color=COLORS["mpc"],     alpha=0.95, label="Risk-MPC")
    ax0.axhline(rho_safe, color=COLORS["safe"], linestyle=(0, (4, 2)),
                linewidth=1.0, label=r"$\rho_{safe}=0.85$")
    for cx, c_peak, m_peak in zip(x, [peak_matched_c, peak_mis_c],
                                       [peak_matched_m, peak_mis_m]):
        reduction = (c_peak - m_peak) / c_peak * 100.0
        ax0.text(cx, max(c_peak, m_peak) + 0.04, f"$-${reduction:.1f}%",
                 ha="center", va="bottom", fontsize=9.0, fontweight="bold",
                 color="#0B4D80")
    ax0.set_xticks(x)
    ax0.set_xticklabels([
        r"matched $\mu=0.8$",
        r"$\mu$ mismatch (plant $0.6$, ctrl $0.8$)",
    ])
    ax0.set_ylabel(r"Peak $\rho_{max}$")
    ax0.set_title("Robustness: Risk-MPC keeps the gain under $\\mu$ mismatch")
    ax0.legend(loc="upper left", fontsize=8.4, ncol=1)
    ax0.set_ylim(0.0, max(peak_matched_c, peak_mis_c) * 1.20)

    # ---- Right: Risk-MPC ρ_max trajectories under matched and mismatched μ ----
    ax1 = axes[1]
    win = (t >= 0.5) & (t <= 4.0)
    ax1.plot(t[win], rho_matched_m[win], color=COLORS["matched"], linewidth=1.8,
             label=r"matched $\mu=0.8/0.8$")
    ax1.plot(t[win], rho_mis_m[win],     color=COLORS["mismatch"], linewidth=1.8,
             label=r"mismatch $\mu=0.6/0.8$")
    ax1.axhline(rho_safe, color=COLORS["safe"], linestyle=(0, (4, 2)),
                linewidth=1.0, label=r"$\rho_{safe}$")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"Risk-MPC $\rho_{max}$")
    ax1.set_title("Graceful degradation under unknown low $\\mu$")
    ax1.legend(loc="upper right", fontsize=8.4)

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
    fig11 = out_dir / "fig11_carsim_mu_mismatch.png"
    make_fig10_carsim_s8(fig10)
    make_fig11_carsim_mu(fig11)

    for fig in (fig10, fig11):
        dst = PAPER_FIG_DIR / fig.name
        shutil.copy2(fig, dst)
        print(f"copied {fig.name} -> {dst}")


if __name__ == "__main__":
    main()
