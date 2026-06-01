from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


CORNER_ORDER = ("FL", "FR", "RL", "RR")
_CORNER_TO_CARSIM = {
    "FL": "L1",
    "FR": "R1",
    "RL": "L2",
    "RR": "R2",
}


@dataclass(frozen=True)
class CarSimFmuSignalMap:
    """Signal names for the CarSim FMU generated for Phase 7."""

    steer: str = "IMP_STEER_SW"
    speed: str = "IMP_SPEED"
    active_force: tuple[str, str, str, str] = (
        "IMP_FZEXL1",
        "IMP_FZEXR1",
        "IMP_FZEXL2",
        "IMP_FZEXR2",
    )
    brake_pressure: tuple[str, str, str, str] = (
        "IMP_PBK_L1",
        "IMP_PBK_R1",
        "IMP_PBK_L2",
        "IMP_PBK_R2",
    )
    mu_x: tuple[str, str, str, str] = (
        "IMP_MUX_L1",
        "IMP_MUX_R1",
        "IMP_MUX_L2",
        "IMP_MUX_R2",
    )
    mu_y: tuple[str, str, str, str] = (
        "IMP_MUY_L1",
        "IMP_MUY_R1",
        "IMP_MUY_L2",
        "IMP_MUY_R2",
    )
    road_height: tuple[str, str, str, str] = (
        "IMP_ZGND_L1",
        "IMP_ZGND_R1",
        "IMP_ZGND_L2",
        "IMP_ZGND_R2",
    )

    fz: tuple[str, str, str, str] = ("Fz_L1", "Fz_R1", "Fz_L2", "Fz_R2")
    fx: tuple[str, str, str, str] = ("Fx_L1", "Fx_R1", "Fx_L2", "Fx_R2")
    fy: tuple[str, str, str, str] = ("Fy_L1", "Fy_R1", "Fy_L2", "Fy_R2")
    wheel_z: tuple[str, str, str, str] = ("Z_L1", "Z_R1", "Z_L2", "Z_R2")
    wheel_vz: tuple[str, str, str, str] = (
        "Vz_WC_L1",
        "Vz_WC_R1",
        "Vz_WC_L2",
        "Vz_WC_R2",
    )
    jounce: tuple[str, str, str, str] = ("Jnc_L1", "Jnc_R1", "Jnc_L2", "Jnc_R2")
    jounce_rate: tuple[str, str, str, str] = (
        "JncR_L1",
        "JncR_R1",
        "JncR_L2",
        "JncR_R2",
    )
    alpha: tuple[str, str, str, str] = (
        "Alpha_L1",
        "Alpha_R1",
        "Alpha_L2",
        "Alpha_R2",
    )
    kappa: tuple[str, str, str, str] = (
        "Kappa_L1",
        "Kappa_R1",
        "Kappa_L2",
        "Kappa_R2",
    )

    body_outputs: tuple[str, ...] = (
        "Xo",
        "Yo",
        "Yaw",
        "Vx",
        "Vy",
        "AVz",
        "Zcg_SM",
        "Roll",
        "Pitch",
        "AVx",
        "AVy",
        "Ax",
        "Ay",
        "Beta",
        "Vz_SM",
        "Az_SM",
    )

    @property
    def input_names(self) -> tuple[str, ...]:
        return (
            self.steer,
            *self.active_force,
            *self.brake_pressure,
            *self.mu_x,
            *self.mu_y,
            self.speed,
            *self.road_height,
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return (
            *self.body_outputs,
            *self.fz,
            *self.fx,
            *self.fy,
            *self.wheel_z,
            *self.wheel_vz,
            *self.jounce,
            *self.jounce_rate,
            *self.alpha,
            *self.kappa,
        )


@dataclass(frozen=True)
class CarSimFmuPlant:
    """Thin open-loop wrapper around the Phase 7 CarSim FMU.

    Public arrays use the project corner order ``FL, FR, RL, RR``. CarSim
    variables use ``L1, R1, L2, R2`` and are mapped internally.
    """

    fmu_path: Path
    signal_map: CarSimFmuSignalMap = CarSimFmuSignalMap()

    def __post_init__(self) -> None:
        path = Path(self.fmu_path)
        if not path.exists():
            raise FileNotFoundError(path)
        object.__setattr__(self, "fmu_path", path)

    def simulate(
        self,
        t: ArrayLike,
        *,
        speed: ArrayLike | float,
        steer: ArrayLike | float = 0.0,
        active_force: ArrayLike | float = 0.0,
        brake_pressure: ArrayLike | float = 0.0,
        road_height: ArrayLike | float = 0.0,
        mu_x: ArrayLike | float = 0.9,
        mu_y: ArrayLike | float = 0.9,
        step_size: float = 0.00025,
        output_interval: float | None = None,
        outputs: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the FMU with sampled inputs.

        ``speed`` follows the CarSim import channel unit configured in the FMU
        dataset. The current dataset uses the same unit as the exported ``Vx``.
        """

        try:
            from fmpy import simulate_fmu
        except ImportError as exc:  # pragma: no cover - depends on local CarSim tooling.
            raise RuntimeError("CarSim FMU simulation requires the optional package 'fmpy'.") from exc

        time = _validate_time(t)
        input_array = self._build_input_array(
            time,
            speed=speed,
            steer=steer,
            active_force=active_force,
            brake_pressure=brake_pressure,
            road_height=road_height,
            mu_x=mu_x,
            mu_y=mu_y,
        )
        requested_outputs = tuple(outputs) if outputs is not None else self.signal_map.output_names
        out = simulate_fmu(
            str(self.fmu_path),
            start_time=float(time[0]),
            stop_time=float(time[-1]),
            step_size=float(step_size),
            output_interval=float(output_interval if output_interval is not None else np.min(np.diff(time))),
            input=input_array,
            output=requested_outputs,
        )
        return {name: np.asarray(out[name], dtype=float) for name in out.dtype.names}

    def _build_input_array(
        self,
        time: np.ndarray,
        *,
        speed: ArrayLike | float,
        steer: ArrayLike | float,
        active_force: ArrayLike | float,
        brake_pressure: ArrayLike | float,
        road_height: ArrayLike | float,
        mu_x: ArrayLike | float,
        mu_y: ArrayLike | float,
    ) -> np.ndarray:
        names = self.signal_map.input_names
        data = np.zeros(len(time), dtype=[("time", float), *[(name, float) for name in names]])
        data["time"] = time
        data[self.signal_map.steer] = _as_scalar_series(steer, len(time), "steer")
        data[self.signal_map.speed] = _as_scalar_series(speed, len(time), "speed")
        for signal, values in (
            (self.signal_map.active_force, active_force),
            (self.signal_map.brake_pressure, brake_pressure),
            (self.signal_map.road_height, road_height),
            (self.signal_map.mu_x, mu_x),
            (self.signal_map.mu_y, mu_y),
        ):
            corner_values = _as_corner_series(values, len(time), signal[0])
            for idx, name in enumerate(signal):
                data[name] = corner_values[:, idx]
        return data

    def collect_corners(self, result: Mapping[str, np.ndarray], signal_names: Sequence[str]) -> np.ndarray:
        return np.column_stack([np.asarray(result[name], dtype=float) for name in signal_names])

    def session(
        self,
        *,
        start_time: float = 0.0,
        stop_time: float | None = None,
        initial_inputs: Mapping[str, float] | None = None,
    ) -> "CarSimFmuSession":
        """Create a low-level Co-Simulation session for closed-loop control."""

        return CarSimFmuSession(
            self,
            start_time=start_time,
            stop_time=stop_time,
            initial_inputs=initial_inputs,
        )


class CarSimFmuSession:
    """Step-by-step FMI 2.0 Co-Simulation session.

    This is intentionally small: it exposes only real-valued input/output access
    needed by the Phase 7 closed-loop CarSim experiments.
    """

    def __init__(
        self,
        plant: CarSimFmuPlant,
        *,
        start_time: float = 0.0,
        stop_time: float | None = None,
        initial_inputs: Mapping[str, float] | None = None,
    ) -> None:
        self.plant = plant
        self.start_time = float(start_time)
        self.stop_time = None if stop_time is None else float(stop_time)
        self.initial_inputs = dict(initial_inputs or {})
        self._unzipdir: str | None = None
        self._fmu = None
        self._vr_by_name: dict[str, int] = {}

    def __enter__(self) -> "CarSimFmuSession":
        try:
            from fmpy import extract, instantiate_fmu, read_model_description
        except ImportError as exc:  # pragma: no cover - depends on local CarSim tooling.
            raise RuntimeError("CarSim FMU stepping requires the optional package 'fmpy'.") from exc

        self._unzipdir = extract(str(self.plant.fmu_path), unzipdir=tempfile.mkdtemp(prefix="carsim_fmu_"))
        model_description = read_model_description(str(self.plant.fmu_path))
        self._vr_by_name = {
            variable.name: int(variable.valueReference)
            for variable in model_description.modelVariables
        }
        self._fmu = instantiate_fmu(
            self._unzipdir,
            model_description,
            fmi_type="CoSimulation",
            visible=False,
            debug_logging=False,
        )
        self._fmu.setupExperiment(startTime=self.start_time, stopTime=self.stop_time)
        if self.initial_inputs:
            self.set_reals(self.initial_inputs)
        self._fmu.enterInitializationMode()
        self._fmu.exitInitializationMode()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._fmu is not None:
                try:
                    self._fmu.terminate()
                finally:
                    self._fmu.freeInstance()
        finally:
            if self._unzipdir is not None:
                shutil.rmtree(self._unzipdir, ignore_errors=True)
            self._fmu = None
            self._unzipdir = None

    def set_reals(self, values: Mapping[str, float]) -> None:
        if self._fmu is None:
            raise RuntimeError("FMU session is not open.")
        names = tuple(values)
        vr = [self._value_reference(name) for name in names]
        self._fmu.setReal(vr, [float(values[name]) for name in names])

    def get_reals(self, names: Sequence[str]) -> dict[str, float]:
        if self._fmu is None:
            raise RuntimeError("FMU session is not open.")
        name_tuple = tuple(names)
        vr = [self._value_reference(name) for name in name_tuple]
        values = self._fmu.getReal(vr)
        return {name: float(value) for name, value in zip(name_tuple, values)}

    def do_step(self, current_time: float, step_size: float) -> None:
        if self._fmu is None:
            raise RuntimeError("FMU session is not open.")
        ok = self._fmu.doStep(
            currentCommunicationPoint=float(current_time),
            communicationStepSize=float(step_size),
        )
        if ok is False:
            raise RuntimeError(f"FMU doStep failed at t={current_time}.")

    def _value_reference(self, name: str) -> int:
        try:
            return self._vr_by_name[name]
        except KeyError as exc:
            raise KeyError(f"FMU variable {name!r} is not available.") from exc


def _validate_time(t: ArrayLike) -> np.ndarray:
    time = np.asarray(t, dtype=float)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("t must be one-dimensional with at least two samples.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("t must be strictly increasing.")
    return time


def _as_scalar_series(values: ArrayLike | float, n_samples: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(n_samples, float(arr))
    if arr.shape == (n_samples,):
        return arr
    raise ValueError(f"{name} must be scalar or shape ({n_samples},), got {arr.shape}.")


def _as_corner_series(values: ArrayLike | float, n_samples: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full((n_samples, 4), float(arr))
    if arr.shape == (4,):
        return np.tile(arr, (n_samples, 1))
    if arr.shape == (n_samples, 4):
        return arr
    raise ValueError(f"{name} must be scalar, shape (4,), or shape ({n_samples}, 4), got {arr.shape}.")
