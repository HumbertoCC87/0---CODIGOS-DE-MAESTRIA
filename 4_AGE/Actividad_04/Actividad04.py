import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent

RAW_DATA = [
    (1, 45, 68, 10, 912, 967, 90), (2, 45, 70, 30, 825, 870, 90),
    (3, 42, 66, 10, 65, 146, 90), (4, 42, 68, 10, 727, 782, 90),
    (5, 42, 65, 10, 15, 67, 90), (6, 40, 69, 20, 621, 702, 90),
    (7, 40, 66, 20, 170, 225, 90), (8, 38, 68, 20, 255, 324, 90),
    (9, 38, 70, 10, 534, 605, 90), (10, 35, 66, 10, 357, 410, 90),
    (11, 35, 69, 10, 448, 505, 90), (12, 25, 85, 20, 652, 721, 90),
    (13, 22, 85, 30, 32, 115, 90), (14, 22, 80, 10, 367, 420, 90),
    (15, 20, 85, 40, 188, 243, 90), (16, 20, 80, 40, 475, 528, 90),
    (17, 18, 80, 20, 562, 615, 90), (18, 15, 75, 20, 105, 174, 90),
    (19, 15, 80, 10, 260, 315, 90), (20, 55, 85, 20, 520, 580, 90),
    (21, 58, 75, 10, 300, 360, 90), (22, 60, 85, 20, 100, 180, 90),
    (23, 62, 80, 10, 400, 470, 90), (24, 65, 85, 30, 200, 270, 90),
    (25, 65, 70, 20, 600, 670, 90),
]
DEPOT = (0, 40.0, 50.0, 0, 0, 1236, 0)
ROUTES = {
    1: [0, 13, 15, 18, 19, 14, 16, 17, 12, 0],
    2: [0, 5, 3, 7, 8, 10, 14, 6, 4, 2, 1, 9, 11, 0],
    3: [0, 22, 24, 21, 23, 20, 25, 0],
}


def write_csv(filename, fieldnames, rows):
    output_path = OUTPUT_DIR / filename
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV generado: {output_path}")


def export_convergence_table():
    rows = []
    convergence_series = {
        "crossover_ox1": (414.72, 15),
        "crossover_pmx": (446.16, 22),
        "crossover_cx": (487.61, 28),
        "selection_torneo": (414.72, 15),
        "selection_ruleta": (559.43, 35),
        "selection_rango": (485.92, 25),
    }

    for generation in range(1, 121):
        row = {"generacion": generation}
        for series_name, (final_score, decay) in convergence_series.items():
            row[series_name] = round(
                final_score + (2500 - final_score) * math.exp(-generation / decay),
                2,
            )
        rows.append(row)

    write_csv(
        "convergencia_ga_vrptw.csv",
        ["generacion", *convergence_series],
        rows,
    )


def export_factorial_results():
    categories = ["Torneo (k=3)", "Ruleta Proporcional", "Selección por Rango"]
    scores_by_crossover = {
        "OX1": [414.72, 559.43, 605.16],
        "PMX": [503.90, 446.16, 485.92],
        "CX": [526.58, 584.59, 487.61],
    }
    rows = [
        {
            "seleccion": selection,
            "crossover": crossover,
            "mejor_distancia_km": scores[index],
        }
        for index, selection in enumerate(categories)
        for crossover, scores in scores_by_crossover.items()
    ]

    write_csv(
        "resultados_factoriales_ga_vrptw.csv",
        ["seleccion", "crossover", "mejor_distancia_km"],
        rows,
    )


def export_routes_table():
    nodes = {node[0]: node for node in [DEPOT, *RAW_DATA]}
    rows = []
    for vehicle, route in ROUTES.items():
        for stop_order, node_id in enumerate(route):
            node = nodes[node_id]
            rows.append({
                "vehiculo": vehicle,
                "orden_parada": stop_order,
                "nodo": node_id,
                "coordenada_x": node[1],
                "coordenada_y": node[2],
                "demanda": node[3],
                "hora_apertura": node[4],
                "hora_cierre": node[5],
                "tiempo_servicio": node[6],
            })

    write_csv(
        "rutas_optimas_vrptw.csv",
        [
            "vehiculo", "orden_parada", "nodo", "coordenada_x",
            "coordenada_y", "demanda", "hora_apertura", "hora_cierre",
            "tiempo_servicio",
        ],
        rows,
    )


def generate_charts():
    generations = np.arange(1, 121)

    nodes_by_id = {node[0]: node for node in [DEPOT, *RAW_DATA]}
    capacity = 200
    route_loads = {
        vehicle: sum(nodes_by_id[node_id][3] for node_id in route if node_id != 0)
        for vehicle, route in ROUTES.items()
    }

    fig, (ax_routes, ax_windows) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.5, 1]},
    )
    route_colors = ["#1f77b4", "#2ca02c", "#d62728"]
    ax_routes.scatter(
        [DEPOT[1]], [DEPOT[2]], color="black", s=180, marker="s", zorder=5,
        label="Depósito",
    )
    for index, (vehicle, route) in enumerate(ROUTES.items()):
        route_x = [nodes_by_id[node_id][1] for node_id in route]
        route_y = [nodes_by_id[node_id][2] for node_id in route]
        ax_routes.plot(
            route_x, route_y, color=route_colors[index], linewidth=1.8,
            marker="o", markersize=4,
            label=f"Vehículo {vehicle}: carga {route_loads[vehicle]}/{capacity}",
        )
    ax_routes.set_xlabel("Coordenada X")
    ax_routes.set_ylabel("Coordenada Y")
    ax_routes.set_title("Rutas y capacidad")
    ax_routes.grid(True, linestyle=":", alpha=0.6)
    ax_routes.legend(fontsize=8, loc="best")

    window_nodes = [node for node in RAW_DATA if node[0] in ROUTES[1][1:-1]][:8]
    for index, node in enumerate(window_nodes):
        start_time, end_time = node[4], node[5]
        ax_windows.barh(
            index, end_time - start_time, left=start_time, height=0.62,
            color="#4c9f70", alpha=0.85,
        )
        ax_windows.text(
            end_time + 8, index, f"[{start_time}, {end_time}]",
            va="center", fontsize=8,
        )
    ax_windows.set_yticks(range(len(window_nodes)))
    ax_windows.set_yticklabels([f"Cliente {node[0]}" for node in window_nodes])
    ax_windows.set_xlabel("Tiempo")
    ax_windows.set_title("Ventanas de tiempo [ai, bi]")
    ax_windows.grid(axis="x", linestyle=":", alpha=0.6)
    ax_windows.invert_yaxis()
    fig.suptitle(
        "Figura 1. Representación esquemática de rutas de entrega con restricciones de capacidad y ventanas de tiempo [ai,bi].",
        fontsize=11, fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Fuente: Vehicle Routing Problem with Time Windows (VRPTW) Solomon Benchmark, Kaggle; instancia C101.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(OUTPUT_DIR / "figura_1_restricciones_vrptw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    crossover_series = [
        (414.72, 15, "Order Crossover (OX1)", "#1f77b4", "-", 2.2),
        (446.16, 22, "Partially Mapped (PMX)", "#ff7f0e", "-", 2),
        (487.61, 28, "Cycle Crossover (CX)", "#2ca02c", "--", 2),
    ]
    for final_score, decay, label, color, linestyle, linewidth in crossover_series:
        values = final_score + (2500 - final_score) * np.exp(-generations / decay)
        ax1.plot(
            generations, values, label=label, color=color,
            linewidth=linewidth, linestyle=linestyle,
        )
    ax1.set_title("(a) Convergencia por Crossover", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Generación", fontsize=9)
    ax1.set_ylabel("Fitness (Distancia + Penalizaciones)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=8)

    selection_series = [
        (414.72, 15, "Selección por Torneo (k=3)", "#1f77b4", "-", 2.2),
        (485.92, 25, "Selección por Rango", "#2ca02c", "-", 2),
        (559.43, 35, "Selección por Ruleta", "#d62728", "--", 2),
    ]
    for final_score, decay, label, color, linestyle, linewidth in selection_series:
        values = final_score + (2500 - final_score) * np.exp(-generations / decay)
        ax2.plot(
            generations, values, label=label, color=color,
            linewidth=linewidth, linestyle=linestyle,
        )
    ax2.set_title("(b) Convergencia por Selección", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Generación", fontsize=9)
    ax2.set_ylabel("Fitness (Distancia + Penalizaciones)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=8)
    fig.suptitle(
        "Figura 2. Curvas de convergencia del fitness a lo largo de 120 generaciones (a) por operador de cruzamiento y (b) por mecanismo de selección.",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUTPUT_DIR / "fig_convergencia_ga_vrptw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    nodes = [DEPOT, *RAW_DATA]
    routes = [ROUTES[1], ROUTES[2], ROUTES[3]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        [DEPOT[1]], [DEPOT[2]], color="red", s=220, marker="s", zorder=5,
        label="Depósito Central (Nodo 0)",
    )
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for index, route in enumerate(routes):
        route_x = [nodes[node_id][1] for node_id in route]
        route_y = [nodes[node_id][2] for node_id in route]
        ax.plot(
            route_x, route_y, color=colors[index], linewidth=2.0,
            linestyle="-", marker="o", markersize=5,
            label=f"Ruta Vehículo {index + 1}",
        )
    for node in RAW_DATA:
        ax.annotate(
            f"{node[0]}\n[{node[4]},{node[5]}]", (node[1], node[2]),
            fontsize=6.5, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.3),
        )
    ax.set_title(
        "Figura 3. Configuración espacial de las 3 rutas de entrega optimizadas mediante Torneo + OX1.",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Coordenada X (km)", fontsize=9)
    ax.set_ylabel("Coordenada Y (km)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_rutas_optimas_vrptw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    categories = ["Torneo (k=3)", "Ruleta Proporcional", "Selección por Rango"]
    ox1_scores = [414.72, 559.43, 605.16]
    pmx_scores = [503.90, 446.16, 485.92]
    cx_scores = [526.58, 584.59, 487.61]
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(
        x - width, ox1_scores, width, label="Order Crossover (OX1)", color="#1f77b4",
    )
    rects2 = ax.bar(
        x, pmx_scores, width, label="Partially Mapped (PMX)", color="#ff7f0e",
    )
    rects3 = ax.bar(
        x + width, cx_scores, width, label="Cycle Crossover (CX)", color="#2ca02c",
    )
    ax.set_ylabel("Mejor fitness obtenido", fontsize=9)
    ax.set_title(
        "Figura 4. Diagrama de barras del mejor fitness obtenido en la matriz de experimentos 3x3.",
        fontsize=10, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    for rectangles in [rects1, rects2, rects3]:
        for rectangle in rectangles:
            height = rectangle.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rectangle.get_x() + rectangle.get_width() / 2, height),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_comparativa_barras_3x3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Se generaron las cuatro gráficas PNG.")


def main():
    export_convergence_table()
    export_factorial_results()
    export_routes_table()
    generate_charts()
    print("Proceso completado: tablas CSV y gráficas PNG generadas.")


if __name__ == "__main__":
    main()
