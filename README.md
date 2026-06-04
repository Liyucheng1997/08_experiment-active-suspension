# Risk-Aware Active Suspension Experiments

Reference implementation and reproduction toolchain for the paper

> **Risk-Aware Active Suspension Control for Tire Friction Margin
> Protection Using Super-Twisting Normal Load Estimation**
> Yucheng Li *et al.*, 2026. *(BibTeX entry: TBD)*

The repository contains the full-car analytical plant, a super-twisting
normal-load fluctuation observer (STO), the risk-aware MPC and one-step QP
controllers, the comfort baselines (Comfort-QP, LQR, Skyhook, passive), the
nine evaluation scenarios, and the scripts that regenerate every figure and
table in Section V of the paper.

The CarSim co-simulation interface (`plants/carsim_fmu.py`) is included for
transparency, but the proprietary CarSim FMU itself is **not** redistributed.
Reproducing Section V-F requires a local CarSim license; everything else
runs from a clean Python install.

---

## 1. Install

Tested on Python 3.10–3.13 (Windows 11 / Ubuntu 22.04).

```bash
git clone https://github.com/Liyucheng1997/risk-aware-active-suspension-experiments.git
cd risk-aware-active-suspension-experiments
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -e .
```

Dependencies (`numpy`, `scipy`, `cvxpy`, `osqp`, `matplotlib`, `pyyaml`,
`fmpy`, `pytest`) are pinned in `pyproject.toml`.

---

## 2. Reproduction quickstart

Run the test suite first — every controller, plant, observer, and scenario
in the paper has a corresponding unit/scenario test:

```bash
pytest tests/
```

The headline analytical-plant experiment (paper Section V-C scenario sweep
across S1–S9) is a single command:

```bash
python -m risk_aware_active_suspension.scenarios.risk_mpc_scenario_sweep
```

Each scenario module under `src/risk_aware_active_suspension/scenarios/`
writes its outputs (CSVs, NPZs, PNGs) to a timestamped directory under
`results/`. Figure-building scripts in `scripts/` then assemble paper-ready
PNGs from those raw results.

> **Path note.** The figure-building scripts in `scripts/` currently hard-code
> the project path and the OneDrive figure-output path. Before running them,
> edit the `PROJECT` and `PAPER_FIG_DIR` constants at the top of each script
> to match your local checkout and target figure folder.

---

## 3. Paper-figure / table mapping

| Paper artefact                      | Scenario / source                                                         | Figure builder                     |
|-------------------------------------|---------------------------------------------------------------------------|------------------------------------|
| Table I (scenarios)                 | hand-edited in LaTeX                                                      | —                                  |
| Table II — MPC vs one-step QP       | `risk_mpc_vs_qp_comparison`                                               | `scripts/fill_table2_mpc_vs_qp.py` |
| Table III — scenario sweep summary  | `risk_mpc_scenario_sweep`                                                 | (LaTeX, numbers from CSV)          |
| Table IV — CarSim summary           | `carsim_final_figure_pack`                                                | (LaTeX, numbers from CSV)          |
| Fig. 1  — friction circle           | —                                                                         | `scripts/make_friction_circle_fig.py` |
| Fig. 3  — STO multi-wheel tracking  | `multiwheel_sto_validation`                                               | `scripts/make_paper_figs_b_e.py`   |
| Fig. 4  — STO error envelope        | `sto_error_bound_calibration`                                             | `scripts/make_paper_figs_b_e.py`   |
| Fig. 5  — ablation safety metrics   | `risk_mpc_ablation`                                                       | `scripts/make_paper_figs_b_e.py`   |
| Fig. 7  — S9 ablation time series   | `risk_mpc_ablation`                                                       | `scripts/make_paper_figs_b_e.py`   |
| Fig. 8  — MPC vs QP on S9           | `risk_mpc_vs_qp_comparison`                                               | `scripts/make_paper_figs_b_e.py`   |
| Fig. 9  — OSQP timing / horizon     | `risk_mpc_solver_benchmark`                                               | `scripts/make_paper_figs_b_e.py`   |
| Fig. 10 — CarSim S8 time series     | `carsim_risk_mpc_validation`, `carsim_final_figure_pack`                  | `scripts/make_paper_figs_carsim.py`|
| Fig. 11 — CarSim μ-mismatch         | `carsim_robustness_validation`, `carsim_final_figure_pack`                | `scripts/make_paper_figs_carsim.py`|

The `results/` directory is intentionally `.gitignore`-d (raw artefacts can
reach hundreds of MB per phase). Running the scenarios above regenerates
everything the figure scripts consume.

---

## 4. CarSim co-simulation (Section V-F)

The CarSim experiments require a local CarSim 2020+ installation with FMU
export and a Pacejka tire dataset matched to a 205/55R16 all-season tire.
The bridge layer in `src/risk_aware_active_suspension/plants/carsim_fmu.py`
loads the exported FMU through `fmpy` and exposes the same interface as the
analytical full-car plant; the closed-loop scenarios are:

- `scenarios/carsim_open_loop_validation.py` — open-loop $F_z$ agreement check
- `scenarios/carsim_sto_validation.py`       — STO on the Pacejka plant
- `scenarios/carsim_risk_mpc_validation.py`  — S4 / S8 closed-loop
- `scenarios/carsim_robustness_validation.py`— μ-mismatch (paper Fig. 11)
- `scenarios/carsim_final_figure_pack.py`    — packages the above for Fig. 10/11

The FMU file (`*.fmu`) is gitignored and **must be provided by the user**
from their own CarSim export. The parameter override scheme used to match
the analytical plant component-by-component is documented inline at the top
of `carsim_fmu.py`.

---

## 5. Repository layout

```
configs/                       # YAML defaults for vehicle / observer / controller
pyproject.toml                 # editable install metadata + deps
scripts/                       # paper figure / table builders
src/risk_aware_active_suspension/
    controllers/               # comfort_qp, lqr, passive, risk_mpc, risk_qp,
                               #   risk_qp_quarter, risk_weights, skyhook
    inputs/                    # road profile (ISO 8608) and maneuver generators
    metrics/                   # signal RMS / RMSE, tire utilisation ρ
    observers/                 # super-twisting normal-load observer (STO)
    plants/                    # quarter, half, full car (analytical) + CarSim FMU
    scenarios/                 # all paper experiments — runnable as modules
    solvers/                   # OSQP harness with warm start + KKT-factor reuse
    utils/                     # config loader, logger, plotting helpers
tests/                         # pytest unit + scenario regression tests
```

---

## 6. License

MIT — see [`LICENSE`](LICENSE).

If this work is useful in your research, please cite the paper above.
