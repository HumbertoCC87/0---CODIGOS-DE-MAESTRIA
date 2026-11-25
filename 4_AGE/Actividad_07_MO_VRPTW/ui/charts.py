"""Executive Plotly charts for optimization, dispatch, and convergence."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

from core.models import Client, Individual, ScenarioResult

PALETTE = {
    "navy": "#0F172A",
    "teal": "#0F766E",
    "blue": "#0284C7",
    "green": "#059669",
    "amber": "#D97706",
    "indigo": "#4F46E5",
    "grid": "#E2E8F0",
    "muted": "#64748B",
}
ROUTE_COLORS = ["#0F766E", "#0284C7", "#7C3AED", "#D97706", "#059669", "#DB2777"]
ACTIVITY_COLORS = {
    "Conducción": "#3B82F6",
    "Servicio/Descarga": "#10B981",
    "Espera ociosa": "#F59E0B",
    "Tardanza SLA": "#EF4444",
}


def _layout_axes(fig: go.Figure) -> None:
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])


def pareto_3d(
    points: np.ndarray,
    knee: int | None = None,
    selected: int | None = None,
    title: str = "Frente de Pareto 3D",
    customdata: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    hovertemplate = (
        "<b>Solución Pareto</b><br>"
        "Costo: $%{x:,.2f}<br>CO₂: %{y:,.2f} kg<br>"
        "Penalización SLA: %{z:,.2f}<br>"
    )
    if customdata is not None:
        hovertemplate += (
            "Vehículos: %{customdata[0]}<br>Distancia: %{customdata[1]:,.2f} km<br>"
            "Cumplimiento SLA: %{customdata[2]:,.2f}%<br>"
            "Desbalance: %{customdata[3]:,.2f} min<extra></extra>"
        )
    else:
        hovertemplate += "<extra></extra>"
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], customdata=customdata,
        mode="markers",
        marker=dict(size=5, color=points[:, 2], colorscale="Viridis", opacity=.88,
                    colorbar=dict(title="SLA")),
        name="Frente 1", hovertemplate=hovertemplate,
    ))
    if knee is not None:
        fig.add_trace(go.Scatter3d(
            x=[points[knee, 0]], y=[points[knee, 1]], z=[points[knee, 2]],
            mode="markers", marker=dict(size=10, color="#F59E0B", symbol="diamond"),
            name="Knee",
        ))
    if selected is not None:
        fig.add_trace(go.Scatter3d(
            x=[points[selected, 0]], y=[points[selected, 1]], z=[points[selected, 2]],
            mode="markers", marker=dict(size=10, color=PALETTE["teal"], symbol="circle"),
            name="Selección inspeccionada",
        ))
    fig.update_layout(
        title=dict(text=title, x=.02, font=dict(color=PALETTE["navy"])),
        height=540,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=0),
        scene=dict(
            xaxis_title="Costo (MXN)",
            yaxis_title="CO₂ (kg)",
            zaxis_title="Penalización SLA",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
            xaxis=dict(backgroundcolor="#FFFFFF", gridcolor=PALETTE["grid"]),
            yaxis=dict(backgroundcolor="#FFFFFF", gridcolor=PALETTE["grid"]),
            zaxis=dict(backgroundcolor="#FFFFFF", gridcolor=PALETTE["grid"]),
        ),
    )
    return fig


def projections(points: np.ndarray) -> list[go.Figure]:
    pairs = ((0, 1, "Costo (MXN)", "CO₂ (kg)"),
             (0, 2, "Costo (MXN)", "Penalización SLA"),
             (1, 2, "CO₂ (kg)", "Penalización SLA"))
    figures = []
    for x_index, y_index, x_label, y_label in pairs:
        fig = go.Figure(go.Scatter(
            x=points[:, x_index], y=points[:, y_index], mode="markers",
            marker=dict(color=points[:, 2], colorscale="Viridis", size=8,
                        colorbar=dict(title="SLA")),
            hovertemplate=f"{x_label}: %{{x:,.2f}}<br>{y_label}: %{{y:,.2f}}<extra></extra>",
        ))
        fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=35, b=10))
        fig.update_xaxes(title=x_label, gridcolor=PALETTE["grid"])
        fig.update_yaxes(title=y_label, gridcolor=PALETTE["grid"])
        figures.append(fig)
    return figures


def radar(profiles: Sequence[tuple[str, Sequence[float]]]) -> go.Figure:
    axes = ["Costo", "Emisiones", "Cumplimiento SLA", "Uso de flota"]
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["teal"]]
    fig = go.Figure()
    for (name, values), color in zip(profiles, colors):
        closed = list(values) + [values[0]]
        fig.add_trace(go.Scatterpolar(
            r=closed, theta=axes + [axes[0]], fill="toself", opacity=.16,
            line=dict(color=color, width=2), name=name,
        ))
    fig.update_layout(
        template="plotly_white", height=360, margin=dict(l=40, r=40, t=35, b=25),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor=PALETTE["grid"])),
        legend=dict(orientation="h", y=-.12),
    )
    return fig


def scenario_comparison(scenarios: Sequence[ScenarioResult]) -> tuple[go.Figure, go.Figure]:
    colors = [PALETTE["teal"], PALETTE["amber"], PALETTE["green"]]
    fig3d = go.Figure()
    fig2d = go.Figure()
    for scenario, color in zip(scenarios, colors):
        points = np.asarray([item.objectives for item in scenario.pareto], dtype=float)
        if points.size == 0:
            continue
        fig3d.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker=dict(size=5, color=color, opacity=.8), name=scenario.name,
            hovertemplate=(
                f"<b>{scenario.name}</b><br>Costo: $%{{x:,.2f}}"
                "<br>CO₂: %{y:,.2f} kg<br>SLA: %{z:,.2f}<extra></extra>"
            ),
        ))
        fig2d.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1], mode="markers",
            marker=dict(size=8, color=color, opacity=.8), name=scenario.name,
            hovertemplate=(
                f"<b>{scenario.name}</b><br>Costo: $%{{x:,.2f}}"
                "<br>CO₂: %{y:,.2f} kg<extra></extra>"
            ),
        ))
    fig3d.update_layout(
        title="Frentes por escenario", template="plotly_white", height=480,
        margin=dict(l=0, r=0, t=45, b=0), legend=dict(orientation="h", y=1.05),
        scene=dict(xaxis_title="Costo (MXN)", yaxis_title="CO₂ (kg)",
                   zaxis_title="Penalización SLA",
                   camera=dict(eye=dict(x=1.55, y=1.55, z=1.15))),
    )
    fig2d.update_layout(
        title="Costo y emisiones", template="plotly_white", height=380,
        margin=dict(l=10, r=10, t=45, b=20), legend=dict(orientation="h", y=1.12),
        xaxis_title="Costo (MXN)", yaxis_title="CO₂ (kg)",
    )
    _layout_axes(fig2d)
    return fig3d, fig2d


def route_map(clients: Sequence[Client], individual: Individual) -> go.Figure:
    fig = go.Figure()
    for route_index, route in enumerate(individual.routes or []):
        sequence = list(route.sequence)
        if not sequence or sequence[0] != 0:
            sequence.insert(0, 0)
        if sequence[-1] != 0:
            sequence.append(0)
        route_color = ROUTE_COLORS[route_index % len(ROUTE_COLORS)]
        x_coords = [clients[node].x for node in sequence]
        y_coords = [clients[node].y for node in sequence]
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode="lines",
            line=dict(color=route_color, width=2.5), name=f"Vehículo {route_index + 1}",
            hoverinfo="skip",
        ))
        stop_ids = [node for node in sequence if node != 0]
        positions = [sequence.index(node) for node in stop_ids]
        arrivals = [
            route.arrival_times[position]
            if position < len(route.arrival_times) else float("nan")
            for position in positions
        ]
        fig.add_trace(go.Scatter(
            x=[clients[node].x for node in stop_ids],
            y=[clients[node].y for node in stop_ids],
            mode="markers",
            marker=dict(
                size=[9 + min(clients[node].demand, 40) * .35 for node in stop_ids],
                color=route_color, opacity=.88, line=dict(color="#FFFFFF", width=1),
            ),
            customdata=[
                [node, clients[node].ready, clients[node].due,
                 clients[node].demand, arrival]
                for node, arrival in zip(stop_ids, arrivals)
            ],
            name=f"Paradas V{route_index + 1}", showlegend=False,
            hovertemplate=(
                "<b>Cliente %{customdata[0]}</b><br>"
                "Ventana: [%{customdata[1]:.0f}, %{customdata[2]:.0f}] min<br>"
                "Demanda: %{customdata[3]:.1f}<br>"
                "Llegada: %{customdata[4]:.1f} min<extra></extra>"
            ),
        ))
    depot = clients[0]
    fig.add_trace(go.Scatter(
        x=[depot.x], y=[depot.y], mode="markers",
        marker=dict(symbol="square", size=17, color=PALETTE["navy"],
                    line=dict(color="#FFFFFF", width=2)),
        name="Depósito", hovertemplate="<b>Depósito central</b><extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", height=500, margin=dict(l=20, r=20, t=80, b=35),
        legend=dict(orientation="h", x=.5, xanchor="center", y=1.14),
        xaxis_title="Coordenada X", yaxis_title="Coordenada Y",
        xaxis=dict(showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], scaleanchor="x", scaleratio=1),
    )
    return fig


def gantt(individual: Individual) -> go.Figure:
    fig = go.Figure()
    seen: set[str] = set()
    for route in individual.routes or []:
        for event in route.timeline:
            kind = str(event["kind"])
            if kind not in ACTIVITY_COLORS:
                continue
            width = max(0.0, float(event["end"]) - float(event["start"]))
            if width <= 0:
                continue
            fig.add_trace(go.Bar(
                x=[width], y=[f"Vehículo {event['vehicle']}"], base=[event["start"]],
                orientation="h", marker=dict(color=ACTIVITY_COLORS[kind]),
                name=kind, legendgroup=kind, showlegend=kind not in seen,
                customdata=[[event["start"]]],
                hovertemplate=(
                    f"<b>{kind}</b><br>%{{y}} · {event['label']}<br>"
                    "Inicio: %{customdata[0]:.1f} min"
                    "<br>Duración: %{x:.1f} min<extra></extra>"
                ),
            ))
            seen.add(kind)
    for kind, color in ACTIVITY_COLORS.items():
        if kind not in seen:
            fig.add_trace(go.Bar(
                x=[None], y=[None], name=kind, legendgroup=kind,
                marker=dict(color=color), showlegend=True, hoverinfo="skip",
            ))
    fig.update_layout(
        template="plotly_white", barmode="overlay", height=420,
        margin=dict(l=20, r=20, t=80, b=35), xaxis_title="Hora operativa (min)",
        legend=dict(orientation="h", x=.5, xanchor="center", y=1.16),
        yaxis=dict(categoryorder="array",
                   categoryarray=[f"Vehículo {i}" for i in range(1, len(individual.routes or []) + 1)]),
    )
    _layout_axes(fig)
    return fig


def convergence_charts(history: dict) -> tuple[go.Figure, go.Figure]:
    generations = history.get("generation", [])
    hv_values = history.get("hv", [])
    hv = go.Figure()
    hv.add_trace(go.Scatter(
        x=generations, y=hv_values, mode="lines+markers", name="Hipervolumen",
        line=dict(color=PALETTE["teal"], width=3, shape="spline"),
        marker=dict(size=7), fill="tozeroy", fillcolor="rgba(15,118,110,.12)",
        hovertemplate="Generación %{x}<br>HV normalizado: %{y:.4f}<extra></extra>",
    ))
    hv.update_layout(
        title="Convergencia del hipervolumen normalizado", template="plotly_white",
        height=360, margin=dict(l=15, r=15, t=55, b=35),
        xaxis_title="Generación", yaxis_title="HV (mayor es mejor)",
    )
    _layout_axes(hv)

    objectives = go.Figure()
    objective_specs = [
        ("best_f1", "Costo", PALETTE["blue"]),
        ("best_f2", "CO₂", PALETTE["green"]),
        ("best_f3", "Penalización SLA", PALETTE["amber"]),
    ]
    for key, label, color in objective_specs:
        values = np.asarray(history.get(key, []), dtype=float)
        if len(values) == 0:
            continue
        span = float(values.max() - values.min())
        normalized = (values - values.min()) / (span if span > 1e-12 else 1.0)
        objectives.add_trace(go.Scatter(
            x=generations[:len(values)], y=normalized, mode="lines+markers",
            name=label, line=dict(color=color, width=2.5),
            hovertemplate=f"{label}<br>Generación %{{x}}<br>Normalizado: %{{y:.3f}}<extra></extra>",
        ))
    objectives.update_layout(
        title="Mínimos por objetivo (0 = mejor)", template="plotly_white",
        height=360, margin=dict(l=15, r=15, t=55, b=35),
        xaxis_title="Generación", yaxis_title="Mínimo normalizado",
        legend=dict(orientation="h", y=1.12),
    )
    _layout_axes(objectives)
    return hv, objectives


def front_size_chart(history: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=history.get("generation", []), y=history.get("front1", []),
        mode="lines+markers", name="Tamaño Frente 1",
        line=dict(color=PALETTE["indigo"], width=2.5),
    ))
    fig.update_layout(
        title="Tamaño del Frente 1", template="plotly_white", height=280,
        xaxis_title="Generación", yaxis_title="Soluciones no dominadas",
    )
    _layout_axes(fig)
    return fig


def convergence(history: dict) -> go.Figure:
    """Compatibility wrapper: returns the front-size convergence chart."""
    return front_size_chart(history)
