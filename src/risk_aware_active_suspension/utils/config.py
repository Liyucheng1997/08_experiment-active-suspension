from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")


@dataclass(frozen=True)
class VehicleParams:
    m: float
    l_f: float
    l_r: float
    h_g: float
    t: float
    k_s: float
    c_s: float
    m_u: float
    k_t: float
    I_x: float
    I_y: float

    @property
    def wheelbase(self) -> float:
        return self.l_f + self.l_r

    @property
    def sprung_mass_per_corner(self) -> float:
        return self.m / 4.0


@dataclass(frozen=True)
class ControllerParams:
    f_max: float = 4000.0
    df_max: float = 20000.0
    rho_safe: float = 0.85
    rho_th: float = 0.75
    k_rho: float = 25.0


@dataclass(frozen=True)
class ObserverParams:
    lambda_1: float = 20.0
    lambda_2: float = 50.0
    epsilon: float = 0.02
    delta_fz_err: float = 100.0


def from_yaml(path: str | Path) -> VehicleParams:
    """Load the default vehicle section from a YAML config file."""
    data = _read_yaml(path)
    if "vehicle" not in data:
        raise KeyError("Expected top-level 'vehicle' section.")
    return dataclass_from_mapping(VehicleParams, data["vehicle"])


def observer_from_yaml(path: str | Path) -> ObserverParams:
    data = _read_yaml(path)
    if "observer" not in data:
        raise KeyError("Expected top-level 'observer' section.")
    return dataclass_from_mapping(ObserverParams, data["observer"])


def dataclass_from_mapping(cls: type[T], data: dict[str, Any]) -> T:
    expected = {field.name for field in fields(cls)}
    actual = set(data)
    extra = actual - expected
    missing = expected - actual
    if extra or missing:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise ValueError(f"Invalid {cls.__name__} fields: {', '.join(details)}")

    coerced = {}
    for field in fields(cls):
        value = data[field.name]
        if field.type in (float, "float"):
            coerced[field.name] = _coerce_float(field.name, value)
        else:
            coerced[field.name] = value
    return cls(**coerced)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping.")
    return data


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
    return float(value)
