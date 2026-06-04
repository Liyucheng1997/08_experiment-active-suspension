from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
PAPER_ROOT = Path(r"F:\latex\12_RiskAware_ActiveSuspension")
FIG_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "tables"

RED = "#C83737"
BLUE = "#1F77B4"
GRAY = "#2B2B2B"
ORANGE = "#E69F00"
GREEN = "#009E73"
LIGHT_RED = "#E9A7A7"
LIGHT_BLUE = "#A9CFE8"


def main() -> None:
    _set_style()
    phase74 = _latest_run("phase-7.4_carsim_risk-mpc-validation_")
    phase75 = _latest_run("phase-7.5_carsim_robustness-validation_")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows74 = _read_rows(phase74 / "phase7_4_summary.csv")
    rows75 = _read_rows(phase75 / "phase7_5_robustness_summary.csv")
    phase6 = _latest_run("phase-6_risk-mpc_scenario-sweep-main_")
    rows6 = _read_rows(phase6 / "phase6_summary.csv")
    raw74 = np.load(phase74 / "raw.npz")
    raw75 = np.load(phase75 / "raw.npz")

    _plot_carsim_s8(FIG_DIR / "fig10_carsim_s8_validation.png", raw74, rows74)
    _plot_carsim_robustness(FIG_DIR / "fig11_carsim_mu_mismatch.png", raw75, rows75)
    _write_carsim_summary_table(TABLE_DIR / "tab_carsim_summary.tex", rows6, rows74)
    _write_carsim_robustness_table(TABLE_DIR / "tab_carsim_robustness.tex", rows75)
    _write_latex_snippet(TABLE_DIR / "carsim_latex_snippet.tex")

    archive = RESULTS_ROOT / "phase-7.6_carsim_paper-assets"
    archive.mkdir(exist_ok=True)
    shutil.copy2(FIG_DIR / "fig10_carsim_s8_validation.png", archive / "fig10_carsim_s8_validation.png")
    shutil.copy2(FIG_DIR / "fig11_carsim_mu_mismatch.png", archive / "fig11_carsim_mu_mismatch.png")
    shutil.copy2(TABLE_DIR / "tab_carsim_summary.tex", archive / "tab_carsim_summary.tex")
    shutil.copy2(TABLE_DIR / "tab_carsim_robustness.tex", archive / "tab_carsim_robustness.tex")
    shutil.copy2(TABLE_DIR / "carsim_latex_snippet.tex", archive / "carsim_latex_snippet.tex")
    print(archive)


def _latest_run(prefix: str) -> Path:
    runs = sorted(
        [p for p in RESULTS_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(prefix)
    return runs[0]


def _set_style() -> None:
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
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.copy()
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(np.pad(signal, pad, mode="edge"), kernel, mode="valid")[: len(signal)]


def _plot_carsim_s8(path: Path, raw: np.lib.npyio.NpzFile, rows: list[dict[str, str]]) -> None:
    scenario = "S8_worst_case"
    row = next(r for r in rows if r["scenario"] == scenario)
    t = raw[f"{scenario}_comfort_qp_time"]
    rho_c = np.max(raw[f"{scenario}_comfort_qp_rho_contact"], axis=1)
    rho_m = np.max(raw[f"{scenario}_risk_mpc_rho_contact"], axis=1)
    win = max(3, int(round(0.05 / np.median(np.diff(t)))))
    rho_c_env = _moving_average(rho_c, win)
    rho_m_env = _moving_average(rho_m, win)

    # The weakest rear wheel gives the most relevant longitudinal-slip trace.
    rear = [2, 3]
    fz_c = raw[f"{scenario}_comfort_qp_fz_true"]
    weakest_rear = rear[int(np.argmin(np.min(fz_c[:, rear], axis=0)))]
    kappa_c = raw[f"{scenario}_comfort_qp_kappa"][:, weakest_rear]
    kappa_m = raw[f"{scenario}_risk_mpc_kappa"][:, weakest_rear]

    fig, axes = plt.subplots(2, 1, figsize=(7.3, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.plot(t, rho_c, color=LIGHT_RED, linewidth=0.9, alpha=0.65)
    ax.plot(t, rho_m, color=LIGHT_BLUE, linewidth=0.9, alpha=0.65)
    ax.plot(t, rho_c_env, color=RED, label="Comfort-QP (50 ms env.)")
    ax.plot(t, rho_m_env, color=BLUE, label="Risk-MPC (50 ms env.)")
    ax.axhline(0.85, color=GRAY, linestyle=(0, (4, 2)), linewidth=1.0, label=r"$\rho_{safe}$")
    c_peak = int(np.argmax(rho_c))
    m_peak = int(np.argmax(rho_m))
    ax.scatter([t[c_peak]], [rho_c[c_peak]], marker="v", color=RED, s=56, zorder=4)
    ax.scatter([t[m_peak]], [rho_m[m_peak]], marker="v", color=BLUE, s=56, zorder=4)
    ax.annotate(
        f"peak={rho_c[c_peak]:.2f}",
        (t[c_peak], rho_c[c_peak]),
        textcoords="offset points",
        xytext=(-8, 6),
        ha="right",
        color=RED,
        fontsize=8.5,
    )
    ax.annotate(
        f"peak={rho_m[m_peak]:.2f}",
        (t[m_peak], rho_m[m_peak]),
        textcoords="offset points",
        xytext=(8, -24),
        ha="left",
        color=BLUE,
        fontsize=8.5,
    )
    ax.set_ylabel(r"$\rho_{max}$")
    ax.set_xlim(0.5, float(t[-1]))
    ax.set_ylim(0.25, max(float(np.max(rho_c)), float(np.max(rho_m))) * 1.15)
    ax.legend(loc="upper right", ncol=2, fontsize=8.2)
    ax.set_title(
        rf"CarSim S8 ($\mu=0.7$): Risk-MPC cuts peak $\rho$ by {100.0 * _f(row, 'rho_reduction_ratio'):.1f}\%"
    )

    ax = axes[1]
    ax.plot(t, kappa_c, color=RED, label=f"Comfort-QP {['FL','FR','RL','RR'][weakest_rear]}")
    ax.plot(t, kappa_m, color=BLUE, label=f"Risk-MPC {['FL','FR','RL','RR'][weakest_rear]}")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_ylabel(r"Longitudinal slip $\kappa$ [-]")
    ax.set_xlabel("Time [s]")
    ax.legend(loc="upper right", fontsize=8.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_carsim_robustness(path: Path, raw: np.lib.npyio.NpzFile, rows: list[dict[str, str]]) -> None:
    cases = [
        ("matched_mu_0p8", "matched 0.8/0.8", BLUE),
        ("mismatch_actual_0p6_assumed_0p8", "mismatch 0.6/0.8", ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    x = np.arange(len(cases))
    width = 0.35
    axes[0].bar(x - width / 2, [_f(r, "comfort_peak_rho") for r in rows], width, color=RED, alpha=0.88, label="Comfort-QP")
    axes[0].bar(x + width / 2, [_f(r, "mpc_peak_rho") for r in rows], width, color=BLUE, alpha=0.92, label="Risk-MPC")
    axes[0].axhline(0.85, color=GRAY, linestyle=(0, (4, 2)), linewidth=1.0)
    axes[0].set_xticks(x, ["matched\n0.8/0.8", "mismatch\n0.6/0.8"])
    axes[0].set_ylabel(r"Peak $\rho_{max}$")
    axes[0].set_title(r"$\mu$ mismatch")
    axes[0].legend(loc="upper left", fontsize=8.2)

    for case, label, color in cases:
        t = raw[f"{case}_risk_mpc_time"]
        rho_trace = np.max(raw[f"{case}_risk_mpc_rho_contact"], axis=1)
        axes[1].plot(t, _moving_average(rho_trace, max(3, int(round(0.05 / np.median(np.diff(t)))))), color=color, label=label)
    axes[1].axhline(0.85, color=GRAY, linestyle=(0, (4, 2)), linewidth=1.0)
    axes[1].set_xlim(0.5, 4.0)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"Risk-MPC $\rho_{max}$")
    axes[1].set_title("Graceful degradation")
    axes[1].legend(loc="upper right", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_carsim_summary_table(path: Path, analytical_rows: list[dict[str, str]], carsim_rows: list[dict[str, str]]) -> None:
    def row_for(rows: list[dict[str, str]], scenario: str) -> dict[str, str]:
        return next(r for r in rows if r["scenario"] == scenario)

    s4_a = row_for(analytical_rows, "S4_classC_cornering")
    s8_a = row_for(analytical_rows, "S8_worst_case")
    s4_c = row_for(carsim_rows, "S4_classC_cornering")
    s8_c = row_for(carsim_rows, "S8_worst_case")

    def table_row(label: str, plant: str, row: dict[str, str], heave_key: str) -> str:
        return (
            rf"{label} & {plant} & {float(row.get('plant_mu', row.get('mu', 'nan'))):.1f} & "
            rf"{_f(row, 'comfort_peak_rho'):.3f} & {_f(row, 'mpc_peak_rho'):.3f} & "
            rf"{100.0*_f(row, 'rho_reduction_ratio'):.1f} & {_f(row, heave_key):.3f}\\"
        )

    text = rf"""\begin{{table}}[t]
\centering
\caption{{Analytical full-car and high-fidelity CarSim validation on the two exported scenarios. $\Delta\rho_{{\max}}=1-\rho_{{\max}}^{{\mathrm{{MPC}}}}/\rho_{{\max}}^{{\mathrm{{Comfort}}}}$.}}
\label{{tab:carsim-summary}}
\renewcommand{{\arraystretch}}{{1.1}}
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{@{{}}llccccc@{{}}}}
\toprule
Scen. & Plant & $\mu$ & Comfort & Risk-MPC & $\Delta\rho_{{\max}}$ [\%] & Heave RMS\\
\midrule
{table_row("S4", "Analytical", s4_a, "mpc_heave_rms")}
{table_row("S4", "CarSim", s4_c, "mpc_heave_rms_m_s2")}
{table_row("S8", "Analytical", s8_a, "mpc_heave_rms")}
{table_row("S8", "CarSim", s8_c, "mpc_heave_rms_m_s2")}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    path.write_text(text, encoding="utf-8")


def _write_carsim_robustness_table(path: Path, rows: list[dict[str, str]]) -> None:
    matched = next(r for r in rows if r["scenario"] == "matched_mu_0p8")
    mismatch = next(r for r in rows if r["scenario"] == "mismatch_actual_0p6_assumed_0p8")
    text = rf"""\begin{{table}}[t]
\centering
\caption{{CarSim friction-mismatch robustness on S8. The mismatch case uses $\mu_\mathrm{{CarSim}}=0.6$ while the controller assumes $\mu_\mathrm{{ctrl}}=0.8$.}}
\label{{tab:carsim-mu-mismatch}}
\renewcommand{{\arraystretch}}{{1.1}}
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
Case & $\mu_\mathrm{{CarSim}}$ & $\mu_\mathrm{{ctrl}}$ & Comfort & Risk-MPC\\
\midrule
matched & 0.8 & 0.8 & {_f(matched, 'comfort_peak_rho'):.3f} & {_f(matched, 'mpc_peak_rho'):.3f}\\
mismatch & 0.6 & 0.8 & {_f(mismatch, 'comfort_peak_rho'):.3f} & {_f(mismatch, 'mpc_peak_rho'):.3f}\\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    path.write_text(text, encoding="utf-8")


def _write_latex_snippet(path: Path) -> None:
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig10_carsim_s8_validation.png}
\caption{High-fidelity CarSim S8 validation. The top panel reports
worst-wheel utilization with a 50 ms envelope; the bottom panel reports
the longitudinal slip ratio of the weakest rear wheel. Risk-MPC reduces
the peak utilization while keeping the slip trace closer to the linear
tire region.}
\label{fig:carsim-s8}
\end{figure}

\input{tables/tab_carsim_summary.tex}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig11_carsim_mu_mismatch.png}
\caption{CarSim robustness to friction mismatch on S8. The mismatch
case sets the CarSim road to $\mu=0.6$ while the controller still assumes
$\mu=0.8$. Risk-MPC degrades gracefully and preserves a lower
$\rho_{\max}$ than Comfort-QP.}
\label{fig:carsim-mu-mismatch}
\end{figure}

\input{tables/tab_carsim_robustness.tex}
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
