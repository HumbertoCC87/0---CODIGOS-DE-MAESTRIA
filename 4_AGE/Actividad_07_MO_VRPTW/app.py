"""Executive Streamlit control tower for Green-VRPTW optimization."""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metrics import (
    WEIGHTS,
    hypervolume_3d,
    knee_point,
    spacing,
    spread,
    topsis,
)
from core.evaluator import VRPTWEvaluator
from core.models import ProblemConfig, ScenarioResult
from core.nsga2 import NSGA2
from core.scenarios import run_scenarios
from data.solomon_loader import load_clients
from ui.charts import (
    convergence_charts,
    front_size_chart,
    gantt,
    pareto_3d,
    projections,
    radar,
    route_map,
    scenario_comparison,
)
from ui.exporter import export_evidence
from ui.styles import apply_styles, render_hero, render_kpi_cards


def _relative_context(value: float, average: float, unit: str = "%") -> str:
    if abs(average) <= 1e-9:
        return f"{value - average:+,.2f} vs. promedio"
    return f"{(value - average) / abs(average) * 100:+.1f}{unit} vs. promedio"


def _objective_table(pareto: list) -> pd.DataFrame:
    return pd.DataFrame({
        "Costo MXN": [item.objectives[0] for item in pareto],
        "CO2 kg": [item.objectives[1] for item in pareto],
        "Penalización SLA": [item.objectives[2] for item in pareto],
        "Cumplimiento SLA %": [item.sla_pct for item in pareto],
        "Vehículos": [item.nvehicles for item in pareto],
        "Distancia km": [item.total_distance for item in pareto],
        "Desbalance min": [item.desbalance for item in pareto],
    })


def _fleet_utilization(individual, capacity: float) -> float:
    if not individual.routes:
        return 0.0
    return min(
        1.0,
        float(np.mean([route.load / max(capacity, 1e-9) for route in individual.routes])),
    )


def _radar_values(individual, points: np.ndarray, capacity: float) -> list[float]:
    spans = np.ptp(points, axis=0)
    normalized_cost = (individual.objectives - points.min(axis=0)) / np.where(
        spans > 1e-12, spans, 1.0
    )
    return [
        float(np.clip(1.0 - normalized_cost[0], 0.0, 1.0)),
        float(np.clip(1.0 - normalized_cost[1], 0.0, 1.0)),
        float(np.clip(individual.sla_pct / 100.0, 0.0, 1.0)),
        _fleet_utilization(individual, capacity),
    ]


def _pareto_column_config() -> dict:
    return {
        "Costo MXN": st.column_config.NumberColumn("Costo (MXN)", format="$%.2f"),
        "CO2 kg": st.column_config.NumberColumn("CO₂ (kg)", format="%.2f"),
        "Penalización SLA": st.column_config.NumberColumn("Penalización SLA", format="%.2f"),
        "Cumplimiento SLA %": st.column_config.NumberColumn("Cumplimiento SLA", format="%.1f%%"),
        "Vehículos": st.column_config.NumberColumn("Vehículos", format="%d"),
        "Distancia km": st.column_config.NumberColumn("Distancia (km)", format="%.2f"),
        "Desbalance min": st.column_config.NumberColumn("Desbalance (min)", format="%.2f"),
    }


st.set_page_config(
    page_title="Green-VRPTW Control Tower",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()

with st.sidebar:
    st.header("Control de despacho")
    instance = st.selectbox("Instancia de ruteo", ["C101", "R101", "RC101"])
    n = st.select_slider("Clientes objetivo", options=[25, 50, 100], value=25)
    run = st.button(
        "Ejecutar Optimización",
        type="primary",
        use_container_width=True,
    )

    with st.expander("📂 Datos y CSV de Kaggle", expanded=False):
        uploaded = st.file_uploader(
            "CSV Solomon (depósito en primera fila)",
            type=["csv"],
            help="El archivo se procesa en memoria y no se copia a outputs/.",
        )

    with st.expander("🚚 Parámetros de Flota y Costos", expanded=False):
        capacity = st.number_input("Capacidad Q", 50.0, 1000.0, 200.0, 25.0)
        fixed = st.number_input("Costo fijo por vehículo", 500.0, 5000.0, 1850.0, 50.0)
        fuel = st.number_input("Precio combustible", 10.0, 60.0, 25.5, .5)
        fleet = st.selectbox("Flota / emisiones", ["Diésel", "Híbrida", "Eléctrica ligera"])
        co2 = {"Diésel": 2.68, "Híbrida": 1.74, "Eléctrica ligera": 1.34}[fleet]
        peak = st.slider("Factor tráfico pico", 1.0, 1.8, 1.35, .05)
        sinuosity = st.slider("Sinuosidad vial", 1.0, 2.0, 1.25, .05)

    with st.expander("🧬 Hiperparámetros NSGA-II", expanded=False):
        pop = st.slider("Población", 40, 150, 80, 10)
        gens = st.slider("Generaciones", 40, 250, 100, 10)
        pc = st.slider("Probabilidad de cruce", .5, 1.0, .85, .05)
        pm = st.slider("Probabilidad de mutación", .05, .6, .20, .05)
        seed = st.number_input("Semilla reproducible", 0, 99999, 42, 1)

cfg = ProblemConfig(
    capacity=float(capacity),
    fixed_cost=float(fixed),
    fuel_price=float(fuel),
    co2_factor=float(co2),
    peak_factor=float(peak),
    sinuosity=float(sinuosity),
    seed=int(seed),
)
clients = load_clients(
    instance,
    n,
    io.BytesIO(uploaded.getvalue()) if uploaded is not None else None,
    int(seed),
)
client_count = len(clients) - 1

if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None
if "scenarios" not in st.session_state:
    st.session_state.scenarios = None

if run:
    progress_slot = st.empty()
    progress = progress_slot.progress(0.0, text="Preparando la optimización…")
    started_at = time.perf_counter()

    def update_progress(generation: int, data: dict) -> None:
        progress.progress(
            min(generation / max(gens, 1), 1.0),
            text=(
                f"Generación {generation}/{gens} · "
                f"Frente 1: {data['front1']} soluciones"
            ),
        )

    try:
        solver = NSGA2(
            client_count,
            pop,
            generations=gens,
            pc=pc,
            pm=pm,
            seed=int(seed),
        )
        pareto, history, _ = solver.run(
            VRPTWEvaluator(clients, cfg),
            update_progress,
        )
        elapsed = time.perf_counter() - started_at
        st.session_state.optimization_result = {
            "pareto": pareto,
            "history": history,
            "elapsed": elapsed,
            "clients": list(clients),
            "config": ProblemConfig(**cfg.__dict__),
            "instance": instance,
        }
        st.session_state.scenarios = None
        st.session_state.selected_solution_index = 0
    finally:
        progress_slot.empty()

result = st.session_state.optimization_result
if result:
    pareto = result["pareto"]
    history = result["history"]
    display_clients = result["clients"]
    result_config = result["config"]
    front_size = len(pareto)
    render_hero(result["instance"], len(display_clients) - 1, result["elapsed"], front_size)
else:
    pareto = []
    history = {}
    display_clients = clients
    result_config = cfg
    render_hero(instance, client_count, None, 0)

tabs = st.tabs([
    "Explorador Pareto y Decisión",
    "Rutas y Operación",
    "Simulador de Escenarios",
    "Convergencia y Exportación",
])

with tabs[0]:
    if not result:
        st.info("Ajusta los parámetros y ejecuta la optimización para analizar el Frente de Pareto.")
    else:
        points = np.asarray([item.objectives for item in pareto], dtype=float)
        st.markdown('<div class="section-heading">Decisión ejecutiva multiobjetivo</div>',
                    unsafe_allow_html=True)
        left, right = st.columns([1, 2.2])
        with left:
            profile = st.selectbox("Perfil TOPSIS", list(WEIGHTS), index=0)
            manual = st.checkbox("Definir pesos manualmente")
            if manual:
                w1 = st.slider("Costo", 0.0, 1.0, .34, .01, key="weight_cost")
                w2 = st.slider("CO₂", 0.0, 1.0, .33, .01, key="weight_co2")
                w3 = st.slider("SLA", 0.0, 1.0, .33, .01, key="weight_sla")
                weights = np.array([w1, w2, w3])
                if weights.sum() <= 1e-12:
                    weights = np.full(3, 1 / 3)
            else:
                weights = WEIGHTS[profile]

            knee = knee_point(pareto)
            selected_by_profile = topsis(pareto, weights)
            economic = topsis(pareto, WEIGHTS["Económico (CFO)"])
            knee_index = next(i for i, item in enumerate(pareto) if item is knee)
            selected_by_profile_index = next(
                i for i, item in enumerate(pareto) if item is selected_by_profile
            )
            if (
                "selected_solution_index" not in st.session_state
                or st.session_state.selected_solution_index >= len(pareto)
            ):
                st.session_state.selected_solution_index = selected_by_profile_index
            selected_index = st.selectbox(
                "Solución para inspección operativa",
                options=list(range(len(pareto))),
                format_func=lambda index: (
                    f"#{index + 1} · ${pareto[index].objectives[0]:,.0f} MXN · "
                    f"{pareto[index].objectives[1]:,.1f} kg CO₂"
                ),
                key="selected_solution_index",
            )
            selected = pareto[selected_index]
            cost_delta = (
                (selected.objectives[0] - economic.objectives[0])
                / max(abs(economic.objectives[0]), 1e-9) * 100
            )
            co2_reduction = (
                (economic.objectives[1] - selected.objectives[1])
                / max(abs(economic.objectives[1]), 1e-9) * 100
            )
            sla_improvement = (
                (economic.objectives[2] - selected.objectives[2])
                / max(abs(economic.objectives[2]), 1e-9) * 100
            )
            st.markdown("#### Trade-off frente al perfil Económico")
            st.info(
                f"Esta solución cambia el costo en **{cost_delta:+.1f}%**, "
                f"reduce CO₂ en **{co2_reduction:+.1f}%** y mejora la "
                f"penalización SLA en **{sla_improvement:+.1f}%**."
            )

        with right:
            customdata = np.asarray([
                [item.nvehicles, item.total_distance, item.sla_pct, item.desbalance]
                for item in pareto
            ])
            st.plotly_chart(
                pareto_3d(points, knee_index, selected_index, customdata=customdata),
                use_container_width=True,
            )

        profiles = [
            ("Económico", topsis(pareto, WEIGHTS["Económico (CFO)"])),
            ("Verde (ESG)", topsis(pareto, WEIGHTS["Verde (ESG)"])),
            ("Servicio (SLA)", topsis(pareto, WEIGHTS["Servicio (SLA)"])),
            ("Balanceado / Knee", knee),
        ]
        radar_profiles = [
            (name, _radar_values(individual, points, result_config.capacity))
            for name, individual in profiles
        ]
        st.markdown("#### Comparación de perfiles de decisión")
        st.plotly_chart(radar(radar_profiles), use_container_width=True)

        with st.expander("Proyecciones 2D y detalle del Frente 1"):
            projection_figures = projections(points)
            projection_columns = st.columns(3)
            for column, figure in zip(projection_columns, projection_figures):
                column.plotly_chart(figure, use_container_width=True)
            st.dataframe(
                _objective_table(pareto),
                hide_index=True,
                use_container_width=True,
                column_config=_pareto_column_config(),
            )

with tabs[1]:
    if not result:
        st.info("Las rutas y métricas operativas aparecerán después de ejecutar el modelo.")
    else:
        selected_index = min(
            int(st.session_state.get("selected_solution_index", 0)),
            len(pareto) - 1,
        )
        individual = pareto[selected_index]
        average_cost = float(np.mean([item.objectives[0] for item in pareto]))
        average_co2 = float(np.mean([item.objectives[1] for item in pareto]))
        average_sla = float(np.mean([item.sla_pct for item in pareto]))
        average_fleet = float(np.mean([item.nvehicles for item in pareto]))
        average_distance = float(np.mean([item.total_distance for item in pareto]))
        average_wait = float(np.mean([item.total_wait for item in pareto]))
        render_kpi_cards([
            ("Costo operativo", f"${individual.objectives[0]:,.2f}",
             _relative_context(individual.objectives[0], average_cost), "#0284C7"),
            ("Emisiones", f"{individual.objectives[1]:,.2f} kg CO₂",
             _relative_context(individual.objectives[1], average_co2), "#059669"),
            ("Cumplimiento SLA", f"{individual.sla_pct:.1f}%",
             f"{individual.sla_pct - average_sla:+.1f} pp vs. promedio", "#D97706"),
            ("Flota", f"{individual.nvehicles} vehículos",
             _relative_context(individual.nvehicles, average_fleet), "#4F46E5"),
            ("Distancia", f"{individual.total_distance:,.2f} km",
             _relative_context(individual.total_distance, average_distance),
             "#0284C7"),
            ("Espera intermedia", f"{individual.total_wait:,.1f} min",
             _relative_context(individual.total_wait, average_wait), "#D97706"),
        ])
        st.markdown("#### Mapa de rutas cerradas al depósito")
        st.plotly_chart(route_map(display_clients, individual), use_container_width=True)
        st.markdown("#### Diagrama de Gantt · despacho inteligente")
        st.plotly_chart(gantt(individual), use_container_width=True)

        rows = []
        for vehicle, route in enumerate(individual.routes or [], 1):
            rows.append({
                "Vehículo": vehicle,
                "Secuencia": " → ".join(map(str, route.sequence)),
                "Carga": route.load,
                "Ocupación %": min(100.0, route.load / max(result_config.capacity, 1e-9) * 100),
                "Distancia km": route.distance,
                "Costo MXN": route.cost,
                "CO2 kg": route.co2,
                "Espera min": route.waiting,
                "Tardanza min": route.late,
            })
        st.markdown("#### Desempeño por vehículo")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Vehículo": st.column_config.NumberColumn("Vehículo", format="%d"),
                "Secuencia": st.column_config.TextColumn("Ruta cerrada"),
                "Carga": st.column_config.NumberColumn("Carga", format="%.1f"),
                "Ocupación %": st.column_config.ProgressColumn(
                    "Ocupación de capacidad", format="%.1f%%", min_value=0, max_value=100,
                ),
                "Distancia km": st.column_config.NumberColumn("Distancia (km)", format="%.2f km"),
                "Costo MXN": st.column_config.NumberColumn("Costo", format="$%.2f"),
                "CO2 kg": st.column_config.NumberColumn("Emisiones (kg)", format="%.2f kg"),
                "Espera min": st.column_config.NumberColumn("Espera (min)", format="%.1f min"),
                "Tardanza min": st.column_config.NumberColumn("Tardanza (min)", format="%.1f min"),
            },
        )

with tabs[2]:
    st.markdown('<div class="section-heading">Comparación industrial de escenarios</div>',
                unsafe_allow_html=True)
    if st.button("Ejecutar 3 escenarios", use_container_width=True):
        with st.spinner("Evaluando operación base, congestión y flota verde…"):
            st.session_state.scenarios = run_scenarios(
                clients, cfg, gens, pop, int(seed)
            )
    scenarios: list[ScenarioResult] | None = st.session_state.scenarios
    if not scenarios:
        st.info("Ejecuta los tres escenarios para comparar sus frentes y su impacto.")
    else:
        scenario_3d, scenario_2d = scenario_comparison(scenarios)
        chart3d, chart2d = st.columns([1.2, 1])
        chart3d.plotly_chart(scenario_3d, use_container_width=True)
        chart2d.plotly_chart(scenario_2d, use_container_width=True)

        baseline = topsis(scenarios[0].pareto, WEIGHTS["Balanceado"])
        impact_cards = []
        impact_colors = ["#0F766E", "#D97706", "#059669"]
        rows = []
        for scenario, color in zip(scenarios, impact_colors):
            points = np.asarray([item.objectives for item in scenario.pareto], dtype=float)
            balanced = topsis(scenario.pareto, WEIGHTS["Balanceado"])
            cost_change = (
                (balanced.objectives[0] - baseline.objectives[0])
                / max(abs(baseline.objectives[0]), 1e-9) * 100
            )
            co2_change = (
                (balanced.objectives[1] - baseline.objectives[1])
                / max(abs(baseline.objectives[1]), 1e-9) * 100
            )
            impact_cards.append((
                scenario.name,
                f"${balanced.objectives[0]:,.0f} MXN",
                f"Costo {cost_change:+.1f}% · CO₂ {co2_change:+.1f}% vs. base",
                color,
            ))
            rows.append({
                "Escenario": scenario.name,
                "Hipervolumen": hypervolume_3d(points, 15000, int(seed)),
                "Spacing": spacing(points),
                "Spread": spread(points),
                "Mejor costo MXN": points[:, 0].min(),
                "Menores emisiones kg": points[:, 1].min(),
                "Costo balanceado MXN": balanced.objectives[0],
                "CO2 balanceado kg": balanced.objectives[1],
            })
        render_kpi_cards(impact_cards)
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Hipervolumen": st.column_config.NumberColumn("Hipervolumen", format="%.4f"),
                "Spacing": st.column_config.NumberColumn("Spacing", format="%.4f"),
                "Spread": st.column_config.NumberColumn("Spread", format="%.4f"),
                "Mejor costo MXN": st.column_config.NumberColumn("Mejor costo", format="$%.2f"),
                "Menores emisiones kg": st.column_config.NumberColumn(
                    "Menores emisiones", format="%.2f kg"
                ),
                "Costo balanceado MXN": st.column_config.NumberColumn(
                    "Costo balanceado", format="$%.2f"
                ),
                "CO2 balanceado kg": st.column_config.NumberColumn(
                    "CO₂ balanceado", format="%.2f kg"
                ),
            },
        )

with tabs[3]:
    if not result:
        st.info("La convergencia estará disponible después de ejecutar la optimización.")
    else:
        hv_chart, objective_chart = convergence_charts(history)
        hv_column, objective_column = st.columns(2)
        hv_column.plotly_chart(hv_chart, use_container_width=True)
        objective_column.plotly_chart(objective_chart, use_container_width=True)
        with st.expander("Evolución del tamaño del Frente 1"):
            st.plotly_chart(front_size_chart(history), use_container_width=True)

        points = np.asarray([item.objectives for item in pareto], dtype=float)
        render_kpi_cards([
            ("Hipervolumen", f"{hypervolume_3d(points, 50000, int(seed)):.4f}",
             "Mayor = mejor cobertura", "#0F766E"),
            ("Spacing", f"{spacing(points):.4f}",
             "Menor = distribución más uniforme", "#0284C7"),
            ("Spread", f"{spread(points):.4f}",
             "Mide la extensión del Frente 1", "#4F46E5"),
        ])
        st.caption(
            "El HV de convergencia usa límites de normalización fijos durante toda "
            "la corrida para hacer comparables las generaciones."
        )
        if st.button("Exportar paquete de evidencias", type="primary"):
            data = _objective_table(pareto).rename(columns={
                "Penalización SLA": "SLA",
                "Cumplimiento SLA %": "SLA %",
                "Desbalance min": "Desbalance",
            })
            summary = {
                "hipervolumen": hypervolume_3d(points, 50000, int(seed)),
                "spacing": spacing(points),
                "spread": spread(points),
                "soluciones_frente_1": len(pareto),
                "generaciones": gens,
                "tiempo_segundos": result["elapsed"],
            }
            files = export_evidence(
                ROOT / "outputs",
                data,
                summary,
                history=history,
                individual=pareto[min(
                    int(st.session_state.get("selected_solution_index", 0)),
                    len(pareto) - 1,
                )],
            )
            st.success("Evidencias guardadas localmente en outputs/:")
            for file_path in files:
                st.code(file_path)
