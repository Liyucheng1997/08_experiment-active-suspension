from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunLogger:
    run_dir: Path

    @classmethod
    def create(cls, root: str | Path, phase: str, step: str, descriptor: str) -> "RunLogger":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_descriptor = descriptor.replace(" ", "_")
        run_dir = Path(root) / f"{phase}_{step}_{safe_descriptor}_{timestamp}"
        (run_dir / "figures").mkdir(parents=True, exist_ok=False)
        (run_dir / "log.txt").write_text("", encoding="utf-8")
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["key", "value"])
        return cls(run_dir=run_dir)

    @classmethod
    def from_existing(cls, run_dir: str | Path) -> "RunLogger":
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "figures").mkdir(exist_ok=True)
        (path / "log.txt").touch(exist_ok=True)
        if not (path / "metrics.csv").exists():
            with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(["key", "value"])
        return cls(run_dir=path)

    def log(self, message: str) -> None:
        with (self.run_dir / "log.txt").open("a", encoding="utf-8") as file:
            file.write(f"{message}\n")

    def log_kv(self, key: str, value: Any) -> None:
        self.log(f"{key}: {value}")
        with (self.run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([key, value])
