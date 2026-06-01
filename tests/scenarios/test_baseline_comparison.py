import csv
import numpy as np

from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.baseline_comparison import (
    SCENARIOS,
    lane_change_a_y,
    run_phase_3_5,
)


def test_lane_change_a_y_profile_zero_outside_window() -> None:
    t = np.linspace(0.0, 12.0, 1201)
    a_y = lane_change_a_y(t, peak_ay=3.0, start=2.0, duration=4.0)
    # Zero before t=2 and after t=6, peaks bounded by amplitude.
    assert np.all(a_y[t < 2.0] == 0.0)
    assert np.all(a_y[t > 6.0] == 0.0)
    assert np.max(np.abs(a_y)) <= 3.0 + 1e-12
    # Lane change integrates to ~zero net lateral velocity change.
    assert abs(np.trapezoid(a_y, t)) < 0.05


def test_full_car_body_acc_input_drives_roll_in_correct_direction(default_vehicle) -> None:
    # Positive a_y must drive phi negative under our convention so that
    # left side compresses (consistent with normal_loads_quasi_static).
    plant = FullCar(default_vehicle)
    t = np.arange(0.0, 4.0, 0.001)
    roads = np.zeros((len(t), 4))
    body_acc = np.column_stack([np.zeros_like(t), 3.0 * np.ones_like(t)])  # constant a_y
    state = np.zeros(14)
    for k in range(len(t) - 1):
        state = plant.step(state, roads[k], 0.001, body_acc=body_acc[k])
    # After 4 s the body has rolled into a steady-state phi < 0.
    assert state[2] < 0.0
    # Left side (corners 0 and 2) compressed: z_u for those corners has
    # moved downward less than body, so suspension stroke is more negative.
    strokes = plant.suspension_strokes(state)
    assert strokes[0] < strokes[1]  # FL more compressed than FR


def test_phase_3_5_csv_has_all_combinations(tmp_path, repo_root) -> None:
    run_dir = run_phase_3_5(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        results_root=tmp_path,
    )
    csv_path = run_dir / "comparison.csv"
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(SCENARIOS) * 4

    # Acceptance ordering on aggregate_rms for every scenario:
    # passive > skyhook > lqr, and qp is within 10% of lqr.
    for spec in SCENARIOS:
        sub = {r["controller"]: float(r["aggregate_rms"]) for r in rows if r["scenario"] == spec.name}
        assert sub["passive"] > sub["skyhook"], spec.name
        assert sub["skyhook"] > sub["lqr"], spec.name
        assert abs(sub["lqr"] - sub["comfort_qp"]) / sub["lqr"] < 0.10, spec.name

    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "acceptance_ordering_passive_gt_skyhook_gt_lqr_approx_qp,1" in metrics


def test_phase_3_5_runner_writes_per_scenario_figures(tmp_path, repo_root) -> None:
    run_dir = run_phase_3_5(
        vehicle_config=repo_root / "configs" / "vehicle_default.yaml",
        controller_config=repo_root / "configs" / "controller_default.yaml",
        results_root=tmp_path,
    )
    for spec in SCENARIOS:
        assert (run_dir / "figures" / f"{spec.name}_heave.png").exists()
        assert (run_dir / "figures" / f"{spec.name}_roll.png").exists()
    assert (run_dir / "figures" / "lane_change_a_y.png").exists()
