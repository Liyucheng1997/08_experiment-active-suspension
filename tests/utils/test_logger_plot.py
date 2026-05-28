import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def test_run_logger_writes_log_and_metrics(tmp_path: Path) -> None:
    logger = RunLogger.from_existing(tmp_path / "run")

    logger.log("started")
    logger.log_kv("rmse", 1.25)

    assert "started" in (logger.run_dir / "log.txt").read_text(encoding="utf-8")
    with (logger.run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows == [["key", "value"], ["rmse", "1.25"]]


def test_plot_timeseries_produces_demo_figure(tmp_path: Path) -> None:
    t = np.linspace(0.0, 1.0, 11)
    path = tmp_path / "figures" / "demo.pdf"

    fig, ax = plot_timeseries(t, np.sin(t), ylabel="z [m]", title="demo", save_path=path)

    assert path.exists()
    assert ax.get_xlabel() == "Time [s]"
    plt.close(fig)
