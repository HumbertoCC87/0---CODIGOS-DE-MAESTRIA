"""Motor de simulación de escenarios industriales comparativos."""
from __future__ import annotations
from typing import List
from .models import ProblemConfig, ScenarioResult, Client
from .evaluator import VRPTWEvaluator
from .nsga2 import NSGA2


def build_scenarios(base:ProblemConfig)->List[tuple[str,ProblemConfig]]:
    diesel=ProblemConfig(**{**base.__dict__,"co2_factor":2.68})
    congestion=ProblemConfig(**{**base.__dict__,"peak_factor":1.45,"fuel_price":base.fuel_price*1.20})
    green=ProblemConfig(**{**base.__dict__,"capacity":base.capacity*.85,"fixed_cost":base.fixed_cost*1.25,"co2_factor":1.474})
    return [("Operación Base",diesel),("Alta Congestión / Combustible",congestion),("Flota Verde / Ligera",green)]


def run_scenarios(clients:List[Client],base:ProblemConfig,n_generations:int,pop_size:int,seed:int=42)->List[ScenarioResult]:
    results=[]
    for idx,(name,cfg) in enumerate(build_scenarios(base)):
        ev=VRPTWEvaluator(clients,cfg)
        solver=NSGA2(len(clients)-1,pop_size,n_generations,seed=seed+idx)
        pareto,history,_=solver.run(ev)
        results.append(ScenarioResult(name,list(pareto),list(pareto),history,cfg))
    return results
