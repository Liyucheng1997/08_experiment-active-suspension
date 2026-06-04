"""Build Section V.B and V.E paper figures from current generated runs."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(r"F:/我的科研工作/05_Risk_Aware_Active_Suspension")
PAPER_FIG_DIR = Path(r"F:/latex/12_RiskAware_ActiveSuspension/figures")

TABLE2_HORIZON = PROJECT / "results/paper_table2_horizon_ablation_current"
PHASE_5_4 = PROJECT / "results/phase-5.4_risk-mpc_warm-start-timing_20260604_142528"

COLORS = {
    "comfort": "#C83737",
    "n1":      "#6F6F6F",
    "mpc":     "#1F77B4",
    "safe":    "#2B2B2B",
    "warm":    "#1F77B4",
    "cold":    "#C83737",
    "budget":  "#2B2B2B",
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
# Figure 8 — Risk-MPC horizon ablation on S9 (Section V.B)
# --------------------------------------------------------------------------
def make_fig08_mpc_vs_qp(out: Path) -> None:
    r = np.load(TABLE2_HORIZON / "raw_s9.npz")
    t = r["t"]
    comfort_rho = r["comfort_rho"].max(axis=1)
    n1_rho      = r["n1_rho"].max(axis=1)
    n10_rho     = r["n10_rho"].max(axis=1)
    n1_fr  = r["n1_forces"][:, 1]   # FR is the bumped wheel under +a_y
    n10_fr = r["n10_forces"][:, 1]
    rho_safe = 0.85
    bump_t = 1.8

    # Show a tight window around the bump.
    win = (t >= 1.7) & (t <= 2.1)

    fig, axes = plt.subplots(
        2, 1, figsize=(7.3, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )

    ax0 = axes[0]
    ax0.plot(t[win], comfort_rho[win], color=COLORS["comfort"], label="Comfort-QP")
    ax0.plot(t[win], n1_rho[win],      color=COLORS["n1"],
             label="Risk-MPC ($N_p=1$)")
    ax0.plot(t[win], n10_rho[win],     color=COLORS["mpc"], linewidth=2.0,
             label="Risk-MPC ($N_p=10$)")
    ax0.axhline(rho_safe, color=COLORS["safe"], linestyle=(0, (4, 2)),
                linewidth=1.0, label=r"$\rho_{safe}=0.85$")
    ax0.axvline(bump_t, color="#777777", linestyle=":", linewidth=0.9)
    ax0.set_ylabel(r"$\rho_{max}$")
    ax0.set_title("S9 bump-during-cornering: preview horizon damps post-impact ripple")
    ax0.legend(loc="upper right", fontsize=8.6, ncol=2)
    ax0.set_ylim(0.45, 1.65)
    ax0.text(
        bump_t + 0.005, 1.57, "bump",
        fontsize=8.6, color="#444444", style="italic", va="top",
    )

    ax1 = axes[1]
    ax1.plot(t[win], n1_fr[win],  color=COLORS["n1"],
             label="$N_p=1$ ($F_{e,FR}$)")
    ax1.plot(t[win], n10_fr[win], color=COLORS["mpc"], linewidth=2.0,
             label="$N_p=10$ ($F_{e,FR}$)")
    ax1.axvline(bump_t, color="#777777", linestyle=":", linewidth=0.9)
    ax1.axhline(0, color="#999999", linewidth=0.7)
    ax1.set_ylabel("Outer-front actuator force [N]")
    ax1.set_xlabel("Time [s]")
    ax1.legend(loc="upper right", fontsize=8.6)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 9 — Solve-time distribution + horizon sweep (Section V.E)
# --------------------------------------------------------------------------
def make_fig09_timing_horizon(out: Path) -> None:
    # Left panel: per-step solve-time histogram on the S9 scenario.
    r = np.load(TABLE2_HORIZON / "raw_s9.npz")
    solve_ms = r["n10_solve_time"] * 1000.0  # to milliseconds
    solve_ms = solve_ms[solve_ms > 0.0]      # drop the unused last sample(s)
    mean_ms = float(np.mean(solve_ms))
    p95_ms  = float(np.percentile(solve_ms, 95))
    p99_ms  = float(np.percentile(solve_ms, 99))
    budget_ms = 5.0  # T_s = 5 ms

    # Right panel: horizon sweep from Phase 5.4 benchmark CSV.
    rows: list[dict[str, float]] = []
    with (PHASE_5_4 / "solver_benchmark.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) for k, v in row.items()})
    horizons = sorted({int(r["horizon"]) for r in rows})
    # Use the OSQP inner-solve P95 (same measurement as the scenario sweep
    # and the histogram on the left). The CSV's `p95_wall_ms` includes the
    # full Python loop overhead that would not exist in a compiled C++
    # deployment and would otherwise contradict the paper's V.E claim.
    p95_cold = [next(r["p95_solve_ms"] for r in rows
                     if int(r["horizon"]) == h and int(r["warm_start"]) == 0)
                for h in horizons]
    p95_warm = [next(r["p95_solve_ms"] for r in rows
                     if int(r["horizon"]) == h and int(r["warm_start"]) == 1)
                for h in horizons]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    ax_l = axes[0]
    # Zoom the histogram around the actual distribution; the T_s budget is
    # an order of magnitude larger and is shown by an annotation, not a line.
    upper = max(p99_ms * 1.6, 1.0)
    bins = np.linspace(0.0, upper, 40)
    ax_l.hist(
        solve_ms.clip(0, upper), bins=bins, color=COLORS["mpc"],
        alpha=0.85, edgecolor="white",
    )
    ax_l.axvline(mean_ms, color="#333333", linestyle="--", linewidth=1.0,
                 label=f"mean = {mean_ms:.2f} ms")
    ax_l.axvline(p95_ms, color="#8C4500", linestyle="-.", linewidth=1.2,
                 label=f"P95 = {p95_ms:.2f} ms")
    ax_l.set_xlabel("Per-step solve time [ms]")
    ax_l.set_ylabel("Sample count")
    ax_l.set_title("S9: solve-time distribution ($N_p{=}10$, warm start)")
    ax_l.set_xlim(0.0, upper)
    ax_l.text(
        upper * 0.97, ax_l.get_ylim()[1] if False else 0.92,
        rf"$T_s = {budget_ms:.0f}$ ms budget",
        ha="right", va="top",
        transform=ax_l.get_xaxis_transform(),
        fontsize=8.4, color="#444444", style="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF7E0",
                  edgecolor="#A0A0A0", linewidth=0.5),
    )
    ax_l.legend(loc="upper right", fontsize=8.4)

    ax_r = axes[1]
    ax_r.plot(horizons, p95_cold, marker="o", color=COLORS["cold"],
              label="Cold start")
    ax_r.plot(horizons, p95_warm, marker="s", color=COLORS["warm"],
              linewidth=2.0, label="Warm start + KKT reuse")
    ax_r.axhline(budget_ms, color=COLORS["budget"], linestyle=(0, (4, 2)),
                 linewidth=1.2, label=f"$T_s = {budget_ms:.0f}$ ms budget")
    ax_r.set_xlabel("Horizon length $N_p$")
    ax_r.set_ylabel("P95 solve time [ms]")
    ax_r.set_title("Horizon sweep: P95 OSQP solve time")
    ax_r.set_xticks(horizons)
    ax_r.legend(loc="upper left", fontsize=8.4)
    ax_r.set_ylim(bottom=0.0, top=max(max(p95_cold), max(p95_warm), budget_ms) * 1.15)
    # Annotate the chosen design point
    h_pick = 10
    idx = horizons.index(h_pick)
    ax_r.annotate(
        f"design point\n$N_p={h_pick}$",
        xy=(h_pick, p95_warm[idx]),
        xytext=(h_pick + 1.5, p95_warm[idx] + (max(p95_cold) - min(p95_warm)) * 0.45),
        fontsize=8.4, color="#0B4D80",
        arrowprops=dict(arrowstyle="->", color="#0B4D80", lw=0.9),
    )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    paper_style()
    out_dir = PROJECT / "results/paper_fig_polish"
    out_dir.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig08 = out_dir / "fig08_mpc_vs_qp.png"
    fig09 = out_dir / "fig09_timing_horizon.png"
    make_fig08_mpc_vs_qp(fig08)
    make_fig09_timing_horizon(fig09)

    for fig in (fig08, fig09):
        dst = PAPER_FIG_DIR / fig.name
        shutil.copy2(fig, dst)
        print(f"copied {fig.name} -> {dst}")


if __name__ == "__main__":
    main()
