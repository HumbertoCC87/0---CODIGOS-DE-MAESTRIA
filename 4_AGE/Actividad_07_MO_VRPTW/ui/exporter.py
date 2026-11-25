"""Export publication-quality tables and figures for local use."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from core.models import Individual
from ui.charts import ACTIVITY_COLORS

NAVY = "#0F172A"
TEAL = "#0F766E"
GRID = "#E2E8F0"
MUTED = "#64748B"


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_evidence(
    output_dir: str | Path,
    pareto_df: pd.DataFrame,
    summary: dict | None = None,
    history: dict | None = None,
    individual: Individual | None = None,
) -> list[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = []

    pareto_csv = output_path / "soluciones_pareto.csv"
    pareto_df.to_csv(pareto_csv, index=False)
    files.append(str(pareto_csv))
    if summary:
        summary_csv = output_path / "resumen_indicadores.csv"
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        files.append(str(summary_csv))

    objective_columns = ["Costo MXN", "CO2 kg", "SLA"]
    available = [column for column in objective_columns if column in pareto_df.columns]
    if not pareto_df.empty and len(available) == 3:
        points = pareto_df[available].to_numpy(dtype=float)
        fig = plt.figure(figsize=(8, 6), dpi=300, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:, 2], cmap="viridis", s=22, alpha=.88)
        ax.set_xlabel("Costo (MXN)", labelpad=10, color=NAVY)
        ax.set_ylabel("CO₂ (kg)", labelpad=10, color=NAVY)
        ax.set_zlabel("Penalización SLA", labelpad=8, color=NAVY)
        ax.view_init(elev=25, azim=35)
        ax.tick_params(colors=MUTED, labelsize=8, pad=2)
        ax.xaxis.pane.set_facecolor((.96, .97, .98, 1))
        ax.yaxis.pane.set_facecolor((.96, .97, .98, 1))
        ax.zaxis.pane.set_facecolor((.96, .97, .98, 1))
        ax.set_title("Frente de Pareto Green-VRPTW", color=NAVY, pad=14)
        fig.tight_layout()
        pareto_png = output_path / "pareto_3d_300dpi.png"
        _save_figure(fig, pareto_png)
        files.append(str(pareto_png))

    if history and history.get("generation") and history.get("hv"):
        generations = np.asarray(history["generation"], dtype=float)
        hv_values = np.asarray(history["hv"], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300, facecolor="white")
        ax.plot(generations, hv_values, color=TEAL, linewidth=2.2, marker="o",
                markersize=3.5, label="Hipervolumen normalizado")
        ax.fill_between(generations, hv_values, color=TEAL, alpha=.12)
        ax.set_xlabel("Generación", color=NAVY)
        ax.set_ylabel("HV normalizado", color=NAVY)
        ax.set_title("Convergencia del hipervolumen", color=NAVY, pad=12)
        ax.grid(True, color=GRID, linewidth=.8)
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False)
        fig.tight_layout()
        convergence_png = output_path / "convergencia_hipervolumen_300dpi.png"
        _save_figure(fig, convergence_png)
        files.append(str(convergence_png))

    if individual and individual.routes:
        activity_order = list(ACTIVITY_COLORS)
        vehicle_count = len(individual.routes)
        fig, ax = plt.subplots(
            figsize=(10, max(3.8, .55 * vehicle_count + 1.5)),
            dpi=300, facecolor="white",
        )
        for route_index, route in enumerate(individual.routes):
            y = vehicle_count - route_index - 1
            for event in route.timeline:
                kind = str(event["kind"])
                if kind not in ACTIVITY_COLORS:
                    continue
                start = float(event["start"])
                width = max(0.0, float(event["end"]) - start)
                if width:
                    ax.barh(y, width, left=start, height=.56,
                            color=ACTIVITY_COLORS[kind], edgecolor="white", linewidth=.3)
        ax.set_yticks(range(vehicle_count))
        ax.set_yticklabels([f"Vehículo {i}" for i in range(vehicle_count, 0, -1)])
        ax.set_xlabel("Hora operativa (min)", color=NAVY)
        ax.set_title("Gantt operativo · despacho inteligente", color=NAVY, pad=12)
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        ax.legend(
            handles=[
                Patch(facecolor=ACTIVITY_COLORS[kind], label=kind)
                for kind in activity_order
            ],
            loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=4, frameon=False,
        )
        fig.tight_layout()
        gantt_png = output_path / "gantt_despacho_inteligente_300dpi.png"
        _save_figure(fig, gantt_png)
        files.append(str(gantt_png))

    return files
