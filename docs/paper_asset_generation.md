# Paper Asset Generation

Use `scripts/make_paper_asset.py` when only figure/table formatting changes.
It reuses the reviewed run data by default and overwrites the corresponding
asset in `F:/latex/12_RiskAware_ActiveSuspension`.

## Common Commands

Generate one figure and copy it to the paper:

```powershell
python scripts\make_paper_asset.py --asset fig01
```

Generate selected figures:

```powershell
python scripts\make_paper_asset.py --asset fig01 --asset fig02 --asset fig08
```

Generate all Phase 6.8 analytical figures:

```powershell
python scripts\make_paper_asset.py --asset phase68
```

Generate all figures:

```powershell
python scripts\make_paper_asset.py --asset figures
```

Generate tables:

```powershell
python scripts\make_paper_asset.py --asset table-scenario
python scripts\make_paper_asset.py --asset table-carsim
```

Preview into `results/paper_asset_work` without copying to the paper:

```powershell
python scripts\make_paper_asset.py --asset fig02 --no-paper-copy
```

Refresh cached headline time-series data:

```powershell
python scripts\make_paper_asset.py --asset fig03 --refresh-cache
```

## Data Sources

Defaults point to the current reviewed runs:

- `phase6-source`: `results/phase-6_risk-mpc_scenario-sweep-main_20260604_104056`
- `ablation-source`: `results/phase-6.6_risk-mpc_component-ablation_20260604_104152`
- `phase55-source`: `results/phase-5.5_risk-mpc_mpc-vs-one-step-qp_20260604_104836`
- `phase54-source`: `results/phase-5.4_risk-mpc_warm-start-timing_20260604_104836`
- `carsim-pack`: `results/paper_carsim_4000N_rate8000_fair_analytical_weights/phase-7.4_carsim_risk-mpc-validation_20260604_153240`

Override a source when you intentionally want to use another run:

```powershell
python scripts\make_paper_asset.py --asset fig01 --phase6-source results\phase-6_risk-mpc_scenario-sweep-main_YYYYMMDD_HHMMSS
```

## Notes

- `fig01`, `fig02`, `fig05`, and `fig06` read CSV summaries only.
- `fig03`, `fig04`, and `fig07` use `results/paper_asset_cache`. If the
  cache is absent, only the needed headline traces are regenerated.
- `table-mpc-vs-qp` is not fully cached by the original experiment. Use
  `scripts/fill_table2_mpc_vs_qp.py` when Table II needs numerical updates.
