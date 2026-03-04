"""
Core models for experiment configuration.

These are the validated models that workers consume.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field, model_validator


class HardwareKind(str, Enum):
    CH_GRID = "chalmers_grid"
    IBM = "ibm"


class CircuitSource(str, Enum):
    RANDOM = "random"
    MQT = "mqt"
    RANDOM_PATTERN = "random_pattern"
    RANDOM_DENSITIES = "random_densities"


class LayoutConfig(BaseModel):
    strategy: str = Field(default="trivial", description="Layout strategy")
    seed: int = Field(description="Layouting seed")
    k: int = Field(ge=1, description="Qubits per group")


class HardwareConfig(BaseModel):
    model_config = {"use_enum_values": True}

    num_qubits: int = Field(gt=0)
    num_cols: Optional[int] = Field(None, gt=0)
    num_rows: Optional[int] = Field(None, gt=0)
    kind: HardwareKind
    id: str
    t1: float
    t2: float
    tsw: float
    virtual_rz: bool = True
    layout: LayoutConfig

    @model_validator(mode="after")
    def validate_grid_dimensions(self):
        if self.kind == HardwareKind.CH_GRID:
            if self.num_cols is None or self.num_rows is None:
                raise ValueError("Grid architecture requires num_cols and num_rows")
            if self.num_cols * self.num_rows != self.num_qubits:
                raise ValueError("Grid dimensions don't match num_qubits")
        return self

    @property
    def hardware_label(self) -> str:
        if self.kind == HardwareKind.CH_GRID:
            return f"{self.kind.title()}_{self.num_rows}x{self.num_cols}"
        elif self.kind == HardwareKind.IBM:
            return f"{self.kind.title()}_{self.id}"

    @property
    def description(self) -> str:
        if self.kind == HardwareKind.CH_GRID:
            return f"{self.kind.title()}_{self.num_rows}x{self.num_cols}_{self.layout.strategy}_k{self.layout.k}"
        elif self.kind == HardwareKind.IBM:
            return f"{self.kind.title()}_{self.id}_{self.num_qubits}_{self.layout.strategy}_k{self.layout.k}"


class CircuitConfig(BaseModel):
    model_config = {"use_enum_values": True}

    source: CircuitSource
    id: str
    num_gates: Optional[int] = None
    num_qubits: Optional[int] = None
    random_weight_1q: Optional[float] = None
    depth: Optional[int] = None
    rho_1: Optional[float] = None
    rho_tot: Optional[float] = None
    connectivity: Optional[Literal["full", "native"]] = None
    seed: int = 0
    optimization_level: int = 2
    needs_translation: bool = False
    needs_routing: bool = True

    @property
    def cache_key(self) -> str:
        return (
            f"{self.id}_{self.num_gates}_{self.num_qubits}_{self.seed}_r1q{self.random_weight_1q}"
        )


class SerializationConfig(BaseModel):
    delay_check: bool = True
    topord_method: str = "prio_two"
    seed: int = 0


class ExperimentConfig(BaseModel):
    """
    Configuration for a single experiment.
    """

    exp_name: str
    exp_id: str
    hardware: HardwareConfig
    circuit: CircuitConfig
    serialization: SerializationConfig
    path_output: Path
    creation_time: str = Field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))

    # Escape hatch for source-specific params that don't fit elsewhere
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_paths(self):
        self.path_output.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_qubit_consistency(self):
        if self.circuit.num_qubits and self.circuit.num_qubits > self.hardware.num_qubits:
            raise ValueError(
                f"Circuit requires {self.circuit.num_qubits} qubits "
                f"but hardware only has {self.hardware.num_qubits}"
            )
        return self


def flatten_config(config: BaseModel) -> dict:
    """Flatten a Pydantic model into dot-notation keys."""

    def flatten(d: dict, parent: str = "") -> dict:
        items = []
        for k, v in d.items():
            key = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                items.extend(flatten(v, key).items())
            elif isinstance(v, Path):
                items.append((key, str(v)))
            else:
                items.append((key, v))
        return dict(items)

    return flatten(config.model_dump())
