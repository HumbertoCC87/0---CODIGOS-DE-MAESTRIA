"""NSGA-II puro para el cromosoma híbrido perm + dos genes reales."""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List
from .models import Individual
from .evaluator import VRPTWEvaluator, evaluate_population


def fast_non_dominated_sort(values: np.ndarray) -> List[List[int]]:
    n=len(values)
    if n==0: return []
    le=np.all(values[:,None,:]<=values[None,:,:],axis=2)
    lt=np.any(values[:,None,:]<values[None,:,:],axis=2)
    dominates=le & lt
    np.fill_diagonal(dominates,False)
    count=dominates.sum(axis=0)
    dominated=[np.flatnonzero(dominates[p]).tolist() for p in range(n)]
    fronts=[]
    remaining=np.flatnonzero(count==0).tolist()
    while remaining:
        fronts.append(remaining)
        nxt=[]
        for p in remaining:
            for q in dominated[p]:
                count[q]-=1
                if count[q]==0: nxt.append(int(q))
        remaining=nxt
    return fronts


def crowding_distance(values: np.ndarray, front: List[int]) -> np.ndarray:
    distances=np.zeros(len(values)); local=np.zeros(len(front))
    if len(front)<=2:
        local[:]=np.inf
    else:
        for m in range(values.shape[1]):
            order=np.argsort(values[front,m]); lo=float(values[front[order[0]],m]); hi=float(values[front[order[-1]],m])
            local[order[0]]=local[order[-1]]=np.inf
            span=hi-lo
            if span>1e-12:
                for k in range(1,len(order)-1):
                    local[order[k]]+=(values[front[order[k+1]],m]-values[front[order[k-1]],m])/span
    distances[front]=local
    return distances


def ox1(a: np.ndarray,b: np.ndarray,rng: np.random.Generator)->np.ndarray:
    n=len(a); i,j=sorted(rng.choice(n,2,replace=False)); child=np.full(n,-1,dtype=int)
    child[i:j+1]=a[i:j+1]; used=set(child[i:j+1]); p=(j+1)%n
    for x in b:
        if int(x) not in used:
            child[p]=x; p=(p+1)%n
    return child


def mutate(ind: Individual,rng: np.random.Generator,swap: float=0.2)->Individual:
    if rng.random()<swap:
        i,j=rng.choice(len(ind.perm),2,replace=False); ind.perm[i],ind.perm[j]=ind.perm[j],ind.perm[i]
    else:
        i,j=sorted(rng.choice(len(ind.perm),2,replace=False)); ind.perm[i:j+1]=ind.perm[i:j+1][::-1]
    ind.alpha_q=float(np.clip(ind.alpha_q+rng.normal(0,.03),.55,1.0))
    ind.max_late_split=float(np.clip(ind.max_late_split+rng.normal(0,5.0),0,90))
    ind.objectives=None
    return ind


def tournament(pop:List[Individual], ranks:np.ndarray, dist:np.ndarray,rng:np.random.Generator)->Individual:
    a,b=rng.choice(len(pop),2,replace=False)
    ra,rb=ranks[a],ranks[b]; da,db=dist[a],dist[b]
    win=a if ra<rb or (ra==rb and da>db) else b
    return pop[int(win)]


class NSGA2:
    def __init__(self,n_clients:int,pop_size:int=80,generations:int=100,pc:float=.85,pm:float|None=None,seed:int=42):
        self.n=n_clients; self.pop_size=pop_size; self.generations=generations
        self.pc=pc; self.pm=.20 if pm is None else pm; self.seed=seed; self.rng=np.random.default_rng(seed)

    def initialize(self,evaluator:VRPTWEvaluator)->List[Individual]:
        pop=[]
        for k in range(self.pop_size):
            if k < int(.2*self.pop_size):
                perm=np.arange(1,self.n+1)
                if k%2==0:
                    perm=perm[np.argsort([evaluator.clients[i].ready for i in perm])]
                else:
                    perm=perm[np.argsort([-evaluator.clients[i].demand for i in perm])]
                self.rng.shuffle(perm)
            else:
                perm=self.rng.permutation(np.arange(1,self.n+1))
            pop.append(Individual(perm,float(self.rng.uniform(.55,1)),float(self.rng.uniform(0,90))))
        evaluate_population(pop,evaluator)
        return pop

    def run(self,evaluator:VRPTWEvaluator,callback:Callable[[int,Dict],None]|None=None)->tuple[List[Individual],Dict[str,List[float|int]],np.ndarray]:
        pop=self.initialize(evaluator)
        history: Dict[str, List[float|int]] = {
            "generation": [], "front1": [], "hv": [],
            "best_f1": [], "best_f2": [], "best_f3": [],
        }
        checkpoints: list[tuple[int, np.ndarray, np.ndarray]] = []

        def record_checkpoint(generation: int) -> None:
            values = np.asarray([ind.objectives for ind in pop], dtype=float)
            fronts = fast_non_dominated_sort(values)
            front_values = values[fronts[0]]
            checkpoints.append((generation, values.copy(), front_values.copy()))
            history["generation"].append(generation)
            history["front1"].append(len(fronts[0]))
            history["best_f1"].append(float(values[:, 0].min()))
            history["best_f2"].append(float(values[:, 1].min()))
            history["best_f3"].append(float(values[:, 2].min()))
            if callback:
                callback(generation, {
                    "generation": generation,
                    "front1": len(fronts[0]),
                    "best_f1": history["best_f1"][-1],
                    "best_f2": history["best_f2"][-1],
                    "best_f3": history["best_f3"][-1],
                })

        record_checkpoint(0)
        for gen in range(1,self.generations+1):
            vals=np.asarray([ind.objectives for ind in pop],dtype=float); fronts=fast_non_dominated_sort(vals)
            ranks=np.empty(len(pop),dtype=int)
            for r,front in enumerate(fronts):
                ranks[front]=r
            dist=crowding_distance(vals, np.concatenate(fronts)) if False else np.zeros(len(pop))
            for r,front in enumerate(fronts):
                dist[front]=crowding_distance(vals,front)[front]
            offspring=[]
            while len(offspring)<self.pop_size:
                p1=tournament(pop,ranks,dist,self.rng); p2=tournament(pop,ranks,dist,self.rng)
                if self.rng.random()<self.pc:
                    c1=ox1(p1.perm,p2.perm,self.rng); c2=ox1(p2.perm,p1.perm,self.rng)
                    aq1=(1-self.rng.beta(2,2))*p1.alpha_q+self.rng.beta(2,2)*p2.alpha_q
                    aq2=(1-self.rng.beta(2,2))*p2.alpha_q+self.rng.beta(2,2)*p1.alpha_q
                    l1=(1-self.rng.beta(2,2))*p1.max_late_split+self.rng.beta(2,2)*p2.max_late_split
                    l2=(1-self.rng.beta(2,2))*p2.max_late_split+self.rng.beta(2,2)*p1.max_late_split
                else:
                    c1=p1.perm.copy(); c2=p2.perm.copy(); aq1=p1.alpha_q; aq2=p2.alpha_q; l1=p1.max_late_split; l2=p2.max_late_split
                c1=mutate(Individual(c1,float(np.clip(aq1,.55,1)),float(np.clip(l1,0,90))),self.rng) if self.rng.random()<self.pm else Individual(c1,float(np.clip(aq1,.55,1)),float(np.clip(l1,0,90)))
                c2=mutate(Individual(c2,float(np.clip(aq2,.55,1)),float(np.clip(l2,0,90))),self.rng) if self.rng.random()<self.pm else Individual(c2,float(np.clip(aq2,.55,1)),float(np.clip(l2,0,90)))
                offspring.extend([c1,c2])
            evaluate_population(offspring[:self.pop_size],evaluator)
            combined=pop+offspring[:self.pop_size]
            cv=np.asarray([ind.objectives for ind in combined],dtype=float)
            cf=fast_non_dominated_sort(cv); cranks=np.empty(len(combined),dtype=int); cd=np.zeros(len(combined))
            new=[]
            for r,front in enumerate(cf):
                cranks[front]=r; cd[front]=crowding_distance(cv,front)[front]
                if len(new)+len(front)<=self.pop_size: new.extend(front)
                else:
                    order=sorted(front,key=lambda idx:cd[idx],reverse=True)
                    new.extend(order[:self.pop_size-len(new)]); break
            pop=[combined[i] for i in new]
            if gen%2==0 or gen==self.generations:
                record_checkpoint(gen)
        vals=np.asarray([ind.objectives for ind in pop],dtype=float); fronts=fast_non_dominated_sort(vals)
        pareto=[pop[i] for i in fronts[0]]
        from .metrics import hypervolume_3d_normalized
        ideal = np.min(np.concatenate([snapshot[1] for snapshot in checkpoints]), axis=0)
        nadir = np.max(np.concatenate([snapshot[1] for snapshot in checkpoints]), axis=0)
        history["hv"] = [
            hypervolume_3d_normalized(
                front_values, ideal, nadir, seed=self.seed + generation
            )
            for generation, _, front_values in checkpoints
        ]
        return pareto,history,vals
