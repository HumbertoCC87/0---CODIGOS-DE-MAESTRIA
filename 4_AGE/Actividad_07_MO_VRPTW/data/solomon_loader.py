"""Carga CSV de Kaggle y generación integrada reproducible tipo Solomon."""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import BinaryIO, List
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.models import Client


def normalize_solomon_csv(path: str | Path | BinaryIO) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    aliases = {
        "id": ["cust_no", "customer", "id", "node"], "x": ["xcoord", "x", "coord_x"],
        "y": ["ycoord", "y", "coord_y"], "demand": ["demand"],
        "ready": ["ready_time", "ready"], "due": ["due_date", "due"],
        "service": ["service_time", "service"]}
    low = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for target, names in aliases.items():
        source = next((low[n] for n in names if n in low), None)
        if source is None:
            raise ValueError(f"Falta columna obligatoria: {names[0]}")
        out[target] = df[source]
    return out


def load_clients(
    instance: str = "C101",
    n: int = 25,
    csv_path: str | Path | BinaryIO | None = None,
    seed: int = 42,
) -> List[Client]:
    if csv_path is not None:
        df = normalize_solomon_csv(csv_path)
        selected = df.iloc[:min(n + 1, len(df))].copy()
        if len(selected) < 2:
            raise ValueError("El CSV debe contener el depósito y al menos un cliente.")
        depot = selected.iloc[0]
    else:
        rng = np.random.default_rng(seed)
        n = min(n, 100)
        # Distribución fiel a C101 en los 50 clientes iniciales y extensiones reproducibles.
        if instance == "C101":
            xs = np.linspace(20, 80, 10)[rng.integers(0, 10, n)]
            ys = np.linspace(20, 80, 10)[rng.integers(0, 10, n)]
        elif instance == "R101":
            xs = rng.uniform(0, 100, n); ys = rng.uniform(0, 100, n)
        else:
            xs = np.where(rng.random(n) < 0.55, rng.normal(30, 12, n), rng.uniform(55, 95, n))
            ys = np.where(rng.random(n) < 0.55, rng.normal(35, 12, n), rng.uniform(55, 95, n))
        xs = np.clip(xs, 0, 100); ys = np.clip(ys, 0, 100)
        # Ventanas realistas, con un conjunto operativo de 480 a 1236 minutos.
        if instance == "C101":
            ready = rng.integers(480, 760, n)
        else:
            ready = rng.integers(300, 720, n)
        due = ready + rng.integers(120, 240, n)
        due = np.minimum(due, 1236)
        service = rng.integers(10, 31, n)
        demand = rng.integers(5, 31, n)
        selected = pd.DataFrame({"id":range(1,n+1),"x":xs,"y":ys,"demand":demand,
                                 "ready":ready,"due":due,"service":service})

    if csv_path is not None:
        rows = selected.iloc[1:n + 1]
        depot_x, depot_y = float(depot.x), float(depot.y)
    else:
        rows = selected
        depot_x, depot_y = 40.0, 50.0

    clients = [Client(0, depot_x, depot_y, 0, 0, 1236, 0, 0, 1.0)]
    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        d = float(r["demand"])
        # cuadrante espacial determina topografía reproducible
        topo = 0.95
        if float(r["x"]) >= 50 and float(r["y"]) >= 50: topo = 1.30
        elif float(r["x"]) >= 50: topo = 1.15
        elif float(r["y"]) >= 50: topo = 1.08
        clients.append(Client(i, float(r["x"]), float(r["y"]), d,
                             float(r["ready"]), float(r["due"]), float(r["service"]),
                             d*15.0, topo))
    return clients
