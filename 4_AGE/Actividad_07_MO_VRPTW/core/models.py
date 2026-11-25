"""Modelos de datos tipados para el problema Green-VRPTW."""
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class ProblemConfig:
    capacity: float = 200.0
    fixed_cost: float = 1850.0
    fuel_price: float = 25.50
    cost_km: float = 3.50
    waiting_cost: float = 25.0
    co2_factor: float = 2.68
    peak_factor: float = 1.35
    sinuosity: float = 1.25
    gamma0: float = 0.22
    gamma1: float = 0.14
    horizon: float = 1236.0
    seed: int = 42


@dataclass
class Client:
    id: int
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float
    weight_kg: float = 0.0
    topo_factor: float = 1.0


@dataclass
class RouteMetrics:
    sequence: List[int]
    arrival_times: List[float]
    distance: float
    load: float
    cost: float
    co2: float
    waiting: float
    late: float
    duration: float
    timeline: List[Dict[str, float | str]] = field(default_factory=list)


@dataclass
class Individual:
    perm: np.ndarray
    alpha_q: float
    max_late_split: float
    objectives: np.ndarray | None = None
    routes: List[RouteMetrics] | None = None


@dataclass
class ScenarioResult:
    name: str
    population: List[Individual]
    pareto: List[Individual]
    history: Dict[str, List[float | int]]
    config: ProblemConfig
