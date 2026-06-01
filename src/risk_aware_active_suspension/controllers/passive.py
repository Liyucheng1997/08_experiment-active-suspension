from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PassiveController:
    """Trivial F_e == 0 controller.

    Provides the same call signature as future active controllers so the
    closed-loop harness can switch implementations without other changes.
    """

    n_actuators: int = 4
    name: str = field(default="passive")

    def __post_init__(self) -> None:
        if self.n_actuators <= 0:
            raise ValueError("n_actuators must be positive.")
        object.__setattr__(self, "_zero", np.zeros(self.n_actuators, dtype=float))

    def reset(self) -> None:
        return None

    def compute(self, state: ArrayLike, *_: Any, **__: Any) -> np.ndarray:
        np.asarray(state, dtype=float)
        return self._zero.copy()
