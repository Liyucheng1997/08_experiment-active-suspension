"""Build the friction-margin "money-shot" figure for the Introduction.

Single-column, side-by-side, minimalist: just the two friction circles,
the same demand arrow, and the ρ readouts. All numerical details
(F_z, μ·F_z, definitions of F_h and ρ_safe) live in the caption to
keep the figure visually clean.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(r"F:/我的科研工作/05_Risk_Aware_Active_Suspension")
PAPER_FIG_DIR = Path(r"F:/latex/12_RiskAware_ActiveSuspension/figures")
OUT = PROJECT / "results/paper_fig_polish/fig_friction_margin.png"

COLOR_COMFORT = "#C83737"
COLOR_ACTIVE  = "#1F77B4"
COLOR_GREY    = "#888888"


def paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 450, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "axes.titlesize": 8.8, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
        "axes.grid": True, "grid.color": "#D8D8D8", "grid.alpha": 0.30,
        "grid.linewidth": 0.4, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.4,
    })


def _draw(ax, F_z_kN: float, demand: tuple[float, float], mu: float,
          color: str, title: str, show_ylabel: bool) -> None:
    r = mu * F_z_kN
    F_c_mag = float(np.hypot(*demand))
    rho = F_c_mag / r
    r_safe = 0.85 * r

    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    # Light fill inside the friction circle
    ax.fill(r * np.cos(theta), r * np.sin(theta),
            color=color, alpha=0.08, zorder=1)
    # Friction-circle boundary in regime color
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            color=color, linewidth=1.8, zorder=3)
    # ρ_safe ring (dashed, grey)
    ax.plot(r_safe * np.cos(theta), r_safe * np.sin(theta),
            color=COLOR_GREY, linewidth=0.8, linestyle=(0, (3, 3)),
            alpha=0.85, zorder=2)

    # Demand vector (same on both panels)
    ax.annotate("", xy=demand, xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="#222222", lw=1.8),
                zorder=4)
    ax.scatter([demand[0]], [demand[1]], color="#222222", s=32,
               zorder=5, edgecolor="white", linewidth=0.7)

    # The single number that matters: big ρ value, top-left of panel
    ax.text(-2.9, 2.7, rf"$\rho={rho:.2f}$",
            fontsize=14, fontweight="bold", color=color,
            ha="left", va="top")

    ax.set_title(title, color=color, fontweight="bold", pad=3)

    LIM = 3.1
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.axhline(0.0, color="#BBBBBB", linewidth=0.5, zorder=0)
    ax.axvline(0.0, color="#BBBBBB", linewidth=0.5, zorder=0)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_xlabel(r"$F_x$ [kN]")
    if show_ylabel:
        ax.set_ylabel(r"$F_y$ [kN]")
    else:
        ax.set_yticklabels([])


def make_friction_margin(out: Path) -> None:
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    # widen the gap so subplot titles don't run into each other
    fig.subplots_adjust(wspace=0.22)

    mu = 0.85
    demand = (-0.55, 1.50)

    _draw(axes[0], F_z_kN=2.00, demand=demand, mu=mu,
          color=COLOR_COMFORT, title="Comfort", show_ylabel=True)
    _draw(axes[1], F_z_kN=3.50, demand=demand, mu=mu,
          color=COLOR_ACTIVE, title="Risk-aware (active)",
          show_ylabel=False)

    fig.tight_layout(w_pad=0.6)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    make_friction_margin(OUT)

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    dst = PAPER_FIG_DIR / "fig_friction_margin.png"
    shutil.copy2(OUT, dst)
    print(f"wrote {OUT}")
    print(f"copied to {dst}")


if __name__ == "__main__":
    main()
