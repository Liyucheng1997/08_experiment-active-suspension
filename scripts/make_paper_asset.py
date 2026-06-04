"""Generate one paper figure/table from existing run data.

Default behavior uses the latest reviewed runs and overwrites the matching
asset under F:/latex/12_RiskAware_ActiveSuspension. It does not rerun the full
scenario sweep. Time-series figures use a small cache; if the cache is absent,
only the required headline case is regenerated and cached.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(r"F:/我的科研工作/05_Risk_Aware_Active_Suspension")
PAPER_ROOT = Path(r"F:/latex/12_RiskAware_ActiveSuspension")
PAPER_FIG_DIR = PAPER_ROOT / "figures"
PAPER_TABLE_DIR = PAPER_ROOT / "tables"

DEFAULT_PHASE6 = PROJECT / "results/phase-6_risk-mpc_scenario-sweep-main_20260604_104056"
DEFAULT_ABLATION = PROJECT / "results/phase-6.6_risk-mpc_component-ablation_20260604_104152"
DEFAULT_TABLE2_HORIZON = PROJECT / "results/paper_table2_horizon_ablation_current"
DEFAULT_PHASE54 = PROJECT / "results/phase-5.4_risk-mpc_warm-start-timing_20260604_104836"
DEFAULT_CARSIM_PACK = PROJECT / (
    "results/paper_carsim_4000N_rate8000_fair_analytical_weights/"
    "phase-7.4_carsim_risk-mpc-validation_20260604_153240"
)
DEFAULT_CACHE = PROJECT / "results/paper_asset_cache"
DEFAULT_OUT = PROJECT / "results/paper_asset_work"

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from risk_aware_active_suspension.scenarios import paper_figure_pack as pfp  # noqa: E402


FIGURE_ASSETS = {
    "fig01", "fig02", "fig03", "fig04", "fig05", "fig06", "fig07",
    "fig08", "fig09", "fig10", "fig11", "fig-friction",
}
TABLE_ASSETS = {"table-scenario", "table-carsim", "table-mpc-vs-qp"}
GROUP_ASSETS = {"all", "figures", "tables", "phase68"}
ALL_ASSETS = sorted(FIGURE_ASSETS | TABLE_ASSETS | GROUP_ASSETS)


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir
    out_fig_dir = out_dir / "figures"
    out_table_dir = out_dir / "tables"
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_table_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    assets = _expand_assets(args.asset)
    written: list[Path] = []
    for asset in assets:
        written.extend(_make_asset(asset, args, out_fig_dir, out_table_dir))

    print("Generated assets:")
    for path in written:
        print(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        action="append",
        choices=ALL_ASSETS,
        required=True,
        help="Asset to generate. Repeat this flag for multiple assets.",
    )
    parser.add_argument("--phase6-source", type=Path, default=DEFAULT_PHASE6)
    parser.add_argument("--ablation-source", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--table2-horizon-source", type=Path, default=DEFAULT_TABLE2_HORIZON)
    parser.add_argument("--phase54-source", type=Path, default=DEFAULT_PHASE54)
    parser.add_argument("--carsim-pack", type=Path, default=DEFAULT_CARSIM_PACK)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="Write only to --out-dir; do not overwrite the LaTeX project.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Regenerate cached time-series data for fig03/fig04/fig07.",
    )
    return parser.parse_args()


def _expand_assets(assets: list[str]) -> list[str]:
    expanded: list[str] = []
    for asset in assets:
        if asset == "all":
            expanded.extend(["fig-friction", *[f"fig{i:02d}" for i in range(1, 12)], *sorted(TABLE_ASSETS)])
        elif asset == "figures":
            expanded.extend(["fig-friction", *[f"fig{i:02d}" for i in range(1, 12)]])
        elif asset == "tables":
            expanded.extend(sorted(TABLE_ASSETS))
        elif asset == "phase68":
            expanded.extend([f"fig{i:02d}" for i in range(1, 8)])
        else:
            expanded.append(asset)
    out: list[str] = []
    for asset in expanded:
        if asset not in out:
            out.append(asset)
    return out


def _make_asset(asset: str, args: argparse.Namespace, fig_dir: Path, table_dir: Path) -> list[Path]:
    if asset in {"fig01", "fig02"}:
        return [_make_phase6_figure(asset, args, fig_dir)]
    if asset in {"fig03", "fig04"}:
        return [_make_headline_timeseries(asset, args, fig_dir)]
    if asset in {"fig05", "fig06"}:
        return [_make_ablation_summary_figure(asset, args, fig_dir)]
    if asset == "fig07":
        return [_make_s9_ablation_timeseries(args, fig_dir)]
    if asset in {"fig08", "fig09"}:
        return [_make_mpc_vs_qp_figure(asset, args, fig_dir)]
    if asset in {"fig10", "fig11"}:
        return [_make_carsim_figure(asset, args, fig_dir)]
    if asset == "fig-friction":
        return [_make_friction_figure(args, fig_dir)]
    if asset == "table-scenario":
        return [_make_scenario_table(args, table_dir)]
    if asset == "table-carsim":
        return [_make_carsim_table(args, table_dir)]
    if asset == "table-mpc-vs-qp":
        return [_make_mpc_vs_qp_table(args, table_dir)]
    raise ValueError(f"unsupported asset: {asset}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _copy_to_paper(path: Path, args: argparse.Namespace) -> None:
    if args.no_paper_copy:
        return
    if path.suffix.lower() == ".png":
        shutil.copy2(path, PAPER_FIG_DIR / path.name)
    elif path.suffix.lower() == ".tex":
        shutil.copy2(path, PAPER_TABLE_DIR / path.name)


def _phase6_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    return _read_rows(args.phase6_source / "phase6_summary.csv")


def _ablation_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    return _read_rows(args.ablation_source / "phase6_6_ablation_summary.csv")


def _make_phase6_figure(asset: str, args: argparse.Namespace, fig_dir: Path) -> Path:
    pfp._set_paper_style()
    rows = _phase6_rows(args)
    path = fig_dir / {
        "fig01": "fig01_scenario_peak_rho.png",
        "fig02": "fig02_safety_comfort_tradeoff.png",
    }[asset]
    if asset == "fig01":
        pfp._plot_main_peak_rho(path, rows)
    else:
        pfp._plot_safety_comfort_tradeoff(path, rows)
    _copy_to_paper(path, args)
    return path


def _make_headline_timeseries(asset: str, args: argparse.Namespace, fig_dir: Path) -> Path:
    pfp._set_paper_style()
    scenario = {"fig03": "S8_worst_case", "fig04": "S9_corner_bump"}[asset]
    run = _load_or_make_headline_cache(args, scenario)
    path = fig_dir / {
        "fig03": "fig03_s8_worst_case_timeseries.png",
        "fig04": "fig04_s9_corner_bump_timeseries.png",
    }[asset]
    if asset == "fig03":
        pfp._plot_s8_timeseries(path, run)
    else:
        pfp._plot_s9_timeseries(path, run)
    _copy_to_paper(path, args)
    return path


def _make_ablation_summary_figure(asset: str, args: argparse.Namespace, fig_dir: Path) -> Path:
    pfp._set_paper_style()
    rows = _ablation_rows(args)
    path = fig_dir / {
        "fig05": "fig05_ablation_safety_metrics.png",
        "fig06": "fig06_ablation_tradeoff.png",
    }[asset]
    if asset == "fig05":
        pfp._plot_ablation_peak_p95(path, rows)
    else:
        pfp._plot_ablation_tradeoff(path, rows)
    _copy_to_paper(path, args)
    return path


def _make_s9_ablation_timeseries(args: argparse.Namespace, fig_dir: Path) -> Path:
    pfp._set_paper_style()
    runs = _load_or_make_s9_ablation_cache(args)
    path = fig_dir / "fig07_s9_ablation_timeseries.png"
    pfp._plot_s9_ablation_timeseries(path, runs)
    _copy_to_paper(path, args)
    return path


def _make_mpc_vs_qp_figure(asset: str, args: argparse.Namespace, fig_dir: Path) -> Path:
    import scripts.make_paper_figs_b_e as be

    be.TABLE2_HORIZON = args.table2_horizon_source
    be.PHASE_5_4 = args.phase54_source
    be.paper_style()
    path = fig_dir / {
        "fig08": "fig08_mpc_vs_qp.png",
        "fig09": "fig09_timing_horizon.png",
    }[asset]
    if asset == "fig08":
        be.make_fig08_mpc_vs_qp(path)
    else:
        be.make_fig09_timing_horizon(path)
    _copy_to_paper(path, args)
    return path


def _make_carsim_figure(asset: str, args: argparse.Namespace, fig_dir: Path) -> Path:
    import scripts.make_paper_figs_carsim as carsim

    carsim.PHASE = args.carsim_pack
    carsim.RAW_74 = args.carsim_pack / "raw.npz"
    carsim.paper_style()
    path = fig_dir / {
        "fig10": "fig10_carsim_s8.png",
        "fig11": "fig11_carsim_s4_s8_summary.png",
    }[asset]
    if asset == "fig10":
        carsim.make_fig10_carsim_s8(path)
    else:
        carsim.make_fig11_carsim_s4_s8_summary(path)
    _copy_to_paper(path, args)
    return path


def _make_friction_figure(args: argparse.Namespace, fig_dir: Path) -> Path:
    import scripts.make_friction_circle_fig as friction

    path = fig_dir / "fig_friction_margin.png"
    friction.make_friction_margin(path)
    _copy_to_paper(path, args)
    return path


def _load_or_make_headline_cache(args: argparse.Namespace, scenario: str) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    cache = args.cache_dir / f"{scenario}_headline_timeseries.npz"
    if cache.exists() and not args.refresh_cache:
        r = np.load(cache)
        return {
            "t": r["t"],
            "comfort": {"rho_contact": r["comfort_rho_contact"], "forces": r["comfort_forces"]},
            "mpc": {"rho_contact": r["mpc_rho_contact"], "forces": r["mpc_forces"]},
        }
    plant, vehicle, lqr, observer_params = _model_stack()
    runs = pfp._rerun_selected_full_cases(plant, vehicle, lqr, observer_params, dt=0.001)
    for name, run in runs.items():
        np.savez_compressed(
            args.cache_dir / f"{name}_headline_timeseries.npz",
            t=run["t"],
            comfort_rho_contact=run["comfort"]["rho_contact"],
            comfort_forces=run["comfort"]["forces"],
            mpc_rho_contact=run["mpc"]["rho_contact"],
            mpc_forces=run["mpc"]["forces"],
        )
    return runs[scenario]


def _load_or_make_s9_ablation_cache(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    cache = args.cache_dir / "S9_ablation_timeseries.npz"
    if cache.exists() and not args.refresh_cache:
        r = np.load(cache)
        out: dict[str, dict[str, np.ndarray]] = {"t": {"value": r["t"]}}
        for key in ["full", "A1_no_sto", "A2_sigma_zero", "A4_no_tightening"]:
            out[key] = {"rho_contact": r[f"{key}_rho_contact"]}
        return out
    plant, vehicle, lqr, observer_params = _model_stack()
    runs = pfp._rerun_s9_ablation_cases(plant, vehicle, lqr, observer_params, dt=0.001)
    np.savez_compressed(
        cache,
        t=runs["t"]["value"],
        full_rho_contact=runs["full"]["rho_contact"],
        A1_no_sto_rho_contact=runs["A1_no_sto"]["rho_contact"],
        A2_sigma_zero_rho_contact=runs["A2_sigma_zero"]["rho_contact"],
        A4_no_tightening_rho_contact=runs["A4_no_tightening"]["rho_contact"],
    )
    return runs


def _model_stack():
    vehicle = pfp.from_yaml(PROJECT / "configs/vehicle_default.yaml")
    lqr = pfp.lqr_from_yaml(PROJECT / "configs/controller_default.yaml")
    observer_params = pfp.observer_from_yaml(PROJECT / "configs/observer_default.yaml")
    plant = pfp.FullCar(vehicle)
    return plant, vehicle, lqr, observer_params


def _make_scenario_table(args: argparse.Namespace, table_dir: Path) -> Path:
    rows = sorted(_phase6_rows(args), key=_scenario_table_sort_key)
    path = table_dir / "tab_scenario_summary.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scenario sweep on the full-car analytical plant. $\Delta\rho_{\max}=1-\rho_{\max}^{\mathrm{MPC}}/\rho_{\max}^{\mathrm{Comfort}}$. The comfort column reports the Risk-MPC heave-RMS change relative to Comfort-QP. Force and P95 solve time are Risk-MPC quantities. Scenario definitions are given in Table~\ref{tab:scenarios}.}",
        r"\label{tab:scenario-summary}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{2.3pt}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Scen.} & \multicolumn{2}{c}{Peak $\rho_{\max}$} & $\Delta\rho_{\max}$ & \multicolumn{1}{c}{$\Delta\bar a_{\ddot z_s}$} & MPC $\max|F_e|$ & P95 \\",
        r"\cmidrule(lr){2-3}",
        r"& Comfort & MPC & [\%] & [m/s$^2$] & [N] & [ms] \\",
        r"\midrule",
    ]
    previous_level: str | None = None
    for row in rows:
        risk_level = row.get("risk_level", "")
        if previous_level is not None and risk_level != previous_level:
            lines.append(r"\midrule")
        if risk_level != previous_level:
            lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{risk_level.capitalize()}}}}}\\")
        previous_level = risk_level
        short = row["scenario"].split("_", 1)[0]
        heave_delta = _format_delta_heave(
            float(row["mpc_heave_rms"]) - float(row["comfort_heave_rms"])
        )
        lines.append(
            rf"{short} & {float(row['comfort_peak_rho']):.3f} & {float(row['mpc_peak_rho']):.3f} & "
            rf"${100.0 * float(row['rho_reduction_ratio']):+.1f}$ & "
            rf"{heave_delta} & "
            rf"{float(row['mpc_max_force_n']):.0f} & {float(row['mpc_p95_solve_time_ms']):.2f} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    _copy_to_paper(path, args)
    return path


def _scenario_table_sort_key(row: dict[str, str]) -> tuple[int, int]:
    risk_order = {"low": 0, "medium": 1, "high": 2}
    scenario = row["scenario"].split("_", 1)[0]
    scenario_num = int(scenario[1:]) if scenario.startswith("S") and scenario[1:].isdigit() else 99
    return risk_order.get(row.get("risk_level", "low"), 0), scenario_num


def _make_carsim_table(args: argparse.Namespace, table_dir: Path) -> Path:
    phase_rows = {r["scenario"]: r for r in _phase6_rows(args)}
    s4 = phase_rows["S4_classC_cornering"]
    s8 = phase_rows["S8_worst_case"]
    path = table_dir / "tab_carsim_summary.tex"
    text = rf"""\begin{{table}}[t]
\centering
\caption{{CarSim validation, organized by experiment purpose. $\Delta\rho_{{\max}}=1-\rho_{{\max}}^{{\mathrm{{MPC}}}}/\rho_{{\max}}^{{\mathrm{{Comfort}}}}$.}}
\label{{tab:carsim-summary}}
\renewcommand{{\arraystretch}}{{1.18}}
\setlength{{\tabcolsep}}{{4.0pt}}
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
Case & $\mu_{{\mathrm{{plant}}}}$ & $\mu_{{\mathrm{{ctrl}}}}$ &
$\rho^{{\mathrm{{C}}}}_{{\max}}$ & $\rho^{{\mathrm{{M}}}}_{{\max}}$ &
$\Delta\rho$ [\%] & P95 [ms] \\
\midrule
\multicolumn{{7}}{{@{{}}l}}{{\textbf{{Block A: Plant cross-validation (matched $\mu$)}}}}\\
S4 -- Analytical             & 0.80 & 0.80 & {float(s4['comfort_peak_rho']):.3f} & {float(s4['mpc_peak_rho']):.3f} & {100.0 * float(s4['rho_reduction_ratio']):+.1f} & {float(s4['mpc_p95_solve_time_ms']):.2f} \\
S4 -- CarSim                 & 0.80 & 0.80 & 0.869 & 0.866 & +0.3 & 1.06 \\
S8 -- Analytical             & 0.70 & 0.70 & {float(s8['comfort_peak_rho']):.3f} & {float(s8['mpc_peak_rho']):.3f} & {100.0 * float(s8['rho_reduction_ratio']):+.1f} & {float(s8['mpc_p95_solve_time_ms']):.2f} \\
S8 -- CarSim                 & 0.70 & 0.70 & 1.609 & 1.067 & +33.6 & 0.60 \\
\midrule
\multicolumn{{7}}{{@{{}}l}}{{\textbf{{Block B: $\mu$-mismatch robustness on S8 (CarSim)}}}}\\
Matched baseline            & 0.80 & 0.80 & 1.408 & 1.184 & +15.9 & 0.94 \\
$\mu$ mismatch              & 0.60 & 0.80 & 1.876 & 1.582 & +15.7 & 0.78 \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    path.write_text(text, encoding="utf-8")
    _copy_to_paper(path, args)
    return path


def _make_mpc_vs_qp_table(args: argparse.Namespace, table_dir: Path) -> Path:
    path = table_dir / "tab_mpc_vs_qp.tex"
    csv_path = args.table2_horizon_source / "table2_horizon_ablation.csv"
    rows = _read_rows(csv_path)
    row_by_scenario = {row["scenario"]: row for row in rows}

    def cell(scenario: str, key: str) -> str:
        return f"{float(row_by_scenario[scenario][key]):.3f}"

    text = rf"""\begin{{table}}[t]
\centering
\caption{{Prediction-horizon ablation on S7--S9. The table reports
peak worst-wheel utilization $\rho_{{\max}}$ and heave-acceleration RMS
$\bar a_{{\ddot z_s}}$ over the maneuver window; the two Risk-MPC
variants differ only in $N_p$.}}
\label{{tab:mpc-vs-qp}}
\renewcommand{{\arraystretch}}{{1.18}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{Comfort-QP}} &
   \multicolumn{{2}}{{c}}{{Risk-MPC ($N_p=1$)}} &
   \multicolumn{{2}}{{c}}{{Risk-MPC ($N_p=10$)}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}
Scen.\ & $\rho_{{\max}}$ & $\bar a_{{\ddot z_s}}$
        & $\rho_{{\max}}$ & $\bar a_{{\ddot z_s}}$
        & $\rho_{{\max}}$ & $\bar a_{{\ddot z_s}}$ \\
\midrule
S7 & {cell("S7", "comfort_peak_rho")} & {cell("S7", "comfort_heave_rms")} & {cell("S7", "n1_peak_rho")} & {cell("S7", "n1_heave_rms")} & \textbf{{{cell("S7", "n10_peak_rho")}}} & {cell("S7", "n10_heave_rms")} \\
S8 & {cell("S8", "comfort_peak_rho")} & {cell("S8", "comfort_heave_rms")} & {cell("S8", "n1_peak_rho")} & {cell("S8", "n1_heave_rms")} & \textbf{{{cell("S8", "n10_peak_rho")}}} & {cell("S8", "n10_heave_rms")} \\
S9 & {cell("S9", "comfort_peak_rho")} & {cell("S9", "comfort_heave_rms")} & {cell("S9", "n1_peak_rho")} & {cell("S9", "n1_heave_rms")} & \textbf{{{cell("S9", "n10_peak_rho")}}} & {cell("S9", "n10_heave_rms")} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    path.write_text(text, encoding="utf-8")
    _copy_to_paper(path, args)
    return path


def _format_heave(value: float) -> str:
    if abs(value) < 5e-5:
        return r"$\approx 0$"
    return f"{value:.3f}"


def _format_delta_heave(value: float) -> str:
    if abs(value) < 5e-5:
        return r"$\approx0$"
    return f"${value:+.3f}$"


def _scenario_table_description(scenario: str) -> str:
    descriptions = {
        "S1_classB_straight": r"B $\times0.5$ $+$ straight",
        "S2_smooth_lane_change": r"flat $+$ single lane change",
        "S3_smooth_jturn": r"flat $+$ moderate J-turn",
        "S4_classC_cornering": r"C $\times0.1$ $+$ steady cornering",
        "S5_classC_lane_change": r"C $\times0.25$ $+$ lane change",
        "S6_emergency_brake": r"flat $+$ emergency brake",
        "S7_brake_cornering": r"flat $+$ brake during cornering",
        "S8_worst_case": r"D $\times0.02$ $+$ brake $+$ cornering",
        "S9_corner_bump": r"flat $+$ cornering $+$ FR bump",
    }
    return descriptions.get(scenario, scenario)


if __name__ == "__main__":
    main()
