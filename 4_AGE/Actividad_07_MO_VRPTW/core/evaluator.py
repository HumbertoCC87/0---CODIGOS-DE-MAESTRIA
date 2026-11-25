"""Matriz de distancias, Split híbrido y evaluación Green-VRPTW."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .models import Client, Individual, ProblemConfig, RouteMetrics


class VRPTWEvaluator:
    def __init__(self, clients: Sequence[Client], config: ProblemConfig):
        self.clients = list(clients)
        self.config = config
        coords = np.array([(client.x, client.y) for client in clients], dtype=float)
        delta = coords[:, None, :] - coords[None, :, :]
        self.distance = np.sqrt((delta * delta).sum(axis=2))

    def traffic(self, time: float) -> float:
        if 480 <= time <= 540 or 1020 <= time <= 1140:
            return self.config.peak_factor
        return 1.0

    def _dispatch_time(self, ready_time: float, road_distance: float) -> float:
        factors = {1.0, float(self.config.peak_factor)}
        candidates = [
            (factor, max(0.0, ready_time - road_distance * factor))
            for factor in factors
        ]
        feasible = [
            departure
            for factor, departure in candidates
            if np.isclose(self.traffic(departure), factor)
        ]
        if feasible:
            return max(feasible)
        return min(
            (departure for _, departure in candidates),
            key=lambda departure: (
                abs(departure + road_distance * self.traffic(departure) - ready_time),
                departure + road_distance * self.traffic(departure) > ready_time,
                -departure,
            ),
        )

    def evaluate(self, ind: Individual) -> Individual:
        perm = ind.perm.tolist()
        routes: List[RouteMetrics] = []
        total_cost = 0.0
        total_co2 = 0.0
        total_sla = 0.0
        total_late = 0.0
        total_wait = 0.0
        sla_ok = 0
        total_distance = 0.0
        pos = 0
        individual_capacity = min(
            self.config.capacity,
            max(
                self.config.capacity * ind.alpha_q,
                max((client.demand for client in self.clients[1:]), default=0.0),
            ),
        )

        while pos < len(perm):
            route_number = len(routes) + 1
            sequence = [0]
            arrival_times: list[float] = []
            timeline: list[dict[str, float | str]] = []
            load = distance = cost = co2 = waiting = late = 0.0
            clock = 0.0
            departure_time = 0.0
            previous = 0

            while pos < len(perm):
                client_id = perm[pos]
                client = self.clients[client_id]
                if load + client.demand > individual_capacity and sequence[-1] != 0:
                    break

                road_distance = (
                    self.distance[previous, client_id] * self.config.sinuosity
                )
                first_stop = len(sequence) == 1
                departure = (
                    self._dispatch_time(client.ready, road_distance)
                    if first_stop
                    else clock
                )
                drive_factor = self.traffic(departure)
                travel_time = road_distance * drive_factor
                actual_arrival = departure + travel_time
                stop_wait = max(0.0, client.ready - actual_arrival)
                service_start = actual_arrival + stop_wait
                stop_late = max(0.0, service_start - client.due)

                if stop_late > ind.max_late_split and not first_stop:
                    break

                waiting_for_objective = 0.0 if first_stop else stop_wait
                if first_stop:
                    departure_time = departure
                payload_before_stop = load
                fuel = road_distance * drive_factor * (
                    self.config.gamma0
                    + self.config.gamma1
                    * (payload_before_stop / max(self.config.capacity, 1e-9))
                )
                stop_cost = (
                    road_distance * self.config.cost_km
                    + fuel * self.config.fuel_price
                    + waiting_for_objective * self.config.waiting_cost / 60.0
                )
                stop_co2 = fuel * self.config.co2_factor * client.topo_factor

                if travel_time > 0:
                    timeline.append({
                        "start": departure,
                        "end": actual_arrival,
                        "kind": "Conducción",
                        "vehicle": route_number,
                        "label": f"{previous} → {client_id}",
                    })
                if waiting_for_objective > 0:
                    timeline.append({
                        "start": actual_arrival,
                        "end": service_start,
                        "kind": "Espera ociosa",
                        "vehicle": route_number,
                        "label": f"Espera en cliente {client_id}",
                    })
                if stop_late > 0:
                    timeline.append({
                        "start": client.due,
                        "end": service_start,
                        "kind": "Tardanza SLA",
                        "vehicle": route_number,
                        "label": f"Tardanza en cliente {client_id}",
                    })
                if client.service > 0:
                    timeline.append({
                        "start": service_start,
                        "end": service_start + client.service,
                        "kind": "Servicio/Descarga",
                        "vehicle": route_number,
                        "label": f"Servicio en cliente {client_id}",
                    })

                sequence.append(client_id)
                arrival_times.append(actual_arrival)
                distance += road_distance
                cost += stop_cost
                co2 += stop_co2
                waiting += waiting_for_objective
                late += stop_late
                total_sla += 3.0 * stop_late + 0.35 * waiting_for_objective
                total_wait += waiting_for_objective
                total_late += stop_late
                sla_ok += int(service_start <= client.due + 1e-9)
                load += client.demand
                pos += 1
                previous = client_id
                clock = service_start + client.service

            if len(sequence) == 1:
                raise ValueError(
                    f"No se pudo asignar el cliente {perm[pos]} a una ruta; "
                    "verifique que su demanda no supere la capacidad del vehículo."
                )

            return_distance = (
                self.distance[previous, 0] * self.config.sinuosity
            )
            return_factor = self.traffic(clock)
            return_time = return_distance * return_factor
            return_arrival = clock + return_time
            return_fuel = return_distance * return_factor * (
                self.config.gamma0
                + self.config.gamma1 * (load / max(self.config.capacity, 1e-9))
            )
            return_topography = self.clients[previous].topo_factor
            cost += (
                return_distance * self.config.cost_km
                + return_fuel * self.config.fuel_price
                + self.config.fixed_cost
            )
            co2 += return_fuel * self.config.co2_factor * return_topography
            distance += return_distance
            sequence.append(0)
            arrival_times.append(return_arrival)
            if return_time > 0:
                timeline.append({
                    "start": clock,
                    "end": return_arrival,
                    "kind": "Conducción",
                    "vehicle": route_number,
                    "label": f"{previous} → 0 · retorno",
                })

            duration = return_arrival - departure_time
            routes.append(RouteMetrics(
                sequence=sequence,
                arrival_times=[departure_time, *arrival_times, return_arrival],
                distance=distance,
                load=load,
                cost=cost,
                co2=co2,
                waiting=waiting,
                late=late,
                duration=duration,
                timeline=timeline,
            ))
            total_cost += cost
            total_co2 += co2
            total_distance += distance

        loads = np.array([route.load for route in routes])
        durations = np.array([route.duration for route in routes])
        ind.routes = routes
        ind.objectives = np.array([total_cost, total_co2, total_sla], dtype=float)
        setattr(ind, "total_distance", float(total_distance))
        setattr(ind, "sla_pct", 100.0 * sla_ok / max(1, len(perm)))
        setattr(ind, "total_late", float(total_late))
        setattr(ind, "total_wait", float(total_wait))
        setattr(ind, "desbalance", float(np.std(durations)))
        setattr(ind, "nvehicles", len(routes))
        setattr(ind, "loads", loads)
        return ind


def evaluate_population(
    population: List[Individual], evaluator: VRPTWEvaluator
) -> np.ndarray:
    values = []
    for individual in population:
        values.append(evaluator.evaluate(individual).objectives)
    return np.asarray(values)
