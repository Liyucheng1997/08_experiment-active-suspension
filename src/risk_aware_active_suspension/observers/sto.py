from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_aware_active_suspension.utils.config import ObserverParams


@dataclass
class STO:
    """Super-twisting observer for one unsprung-velocity channel.

    The observer estimates a residual acceleration ``chi`` in
    ``dot(v_u) = phi_known + chi``. The residual force estimate is
    ``d_z_hat = m_u * chi_hat``.
    """

    params: ObserverParams
    unsprung_mass: float
    v_u_hat: float = 0.0
    chi_hat: float = 0.0
    use_saturation: bool = False

    def reset(self, v_u_hat: float = 0.0, chi_hat: float = 0.0) -> None:
        self.v_u_hat = float(v_u_hat)
        self.chi_hat = float(chi_hat)

    def step(self, v_u_meas: float, phi_known: float, dt: float, f_z_bar: float = 0.0) -> tuple[float, float]:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.unsprung_mass <= 0.0:
            raise ValueError("unsprung_mass must be positive.")

        error = self.v_u_hat - float(v_u_meas)
        switching = self._switching(error)
        self.v_u_hat += dt * (
            float(phi_known)
            - self.params.lambda_1 * np.sqrt(abs(error)) * switching
            + self.chi_hat
        )
        self.chi_hat += dt * (-self.params.lambda_2 * switching)

        d_z_hat = self.unsprung_mass * self.chi_hat
        f_z_hat = float(f_z_bar) + d_z_hat
        return float(d_z_hat), float(f_z_hat)

    def _switching(self, error: float) -> float:
        if self.use_saturation:
            if self.params.epsilon <= 0.0:
                raise ValueError("epsilon must be positive when saturation is enabled.")
            return float(np.clip(error / self.params.epsilon, -1.0, 1.0))
        if error > 0.0:
            return 1.0
        if error < 0.0:
            return -1.0
        return 0.0
