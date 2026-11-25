"""Métricas de calidad, Hipervolumen determinista y decisión multicriterio."""
import numpy as np
from typing import Sequence
from .models import Individual

WEIGHTS={"Balanceado":np.array([.34,.33,.33]),"Económico (CFO)":np.array([.60,.15,.25]),
         "Verde (ESG)":np.array([.20,.60,.20]),"Servicio (SLA)":np.array([.20,.15,.65])}

def normalize_points(points:np.ndarray)->np.ndarray:
    lo=points.min(0); hi=points.max(0); span=hi-lo
    return (points-lo)/np.where(span>1e-12,span,1)

def hypervolume_3d(points:np.ndarray,samples:int=50000,seed:int=42,reference:np.ndarray|None=None)->float:
    """Hipervolumen normalizado con muestreo Monte Carlo determinista."""
    if len(points)==0: return 0.0
    norm=normalize_points(points)
    ref=np.ones(3) if reference is None else reference
    rng=np.random.default_rng(seed); samples3=rng.random((samples,3))*ref
    dominated=np.zeros(samples,dtype=bool)
    for p in norm:
        dominated |= np.all(samples3>=p,axis=1)
    # convención: volumen de la caja dominada por puntos minimizadores
    return float(dominated.mean())

def hypervolume_3d_normalized(
    points: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    samples: int = 15000,
    seed: int = 42,
) -> float:
    """Calcula hipervolumen con límites fijos para comparar generaciones."""
    if len(points) == 0:
        return 0.0
    span = np.asarray(nadir, dtype=float) - np.asarray(ideal, dtype=float)
    span = np.where(span > 1e-12, span, 1.0)
    normalized = np.clip((np.asarray(points, dtype=float) - ideal) / span, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    samples3 = rng.random((samples, 3))
    dominated = np.zeros(samples, dtype=bool)
    for point in normalized:
        dominated |= np.all(samples3 >= point, axis=1)
    return float(dominated.mean())

def spacing(points:np.ndarray)->float:
    if len(points)<2: return 0.0
    norm=normalize_points(points); d=[]
    for i,p in enumerate(norm):
        other=np.delete(norm,i,axis=0)
        d.append(np.linalg.norm(other-p,axis=1).min())
    d=np.asarray(d); return float(np.sqrt(np.mean((d-d.mean())**2)))

def spread(points:np.ndarray)->float:
    if len(points)<3: return 0.0
    norm=normalize_points(points); center=norm.mean(0)
    return float(np.mean(np.linalg.norm(norm-center,axis=1)))

def knee_point(population:Sequence[Individual])->Individual:
    points=np.asarray([p.objectives for p in population]); norm=normalize_points(points)
    idx=int(np.argmin(np.linalg.norm(norm-norm.min(0),axis=1)))
    return population[idx]

def topsis(population:Sequence[Individual],weights:np.ndarray)->Individual:
    points=np.asarray([p.objectives for p in population],dtype=float); norm=normalize_points(points)
    w=np.asarray(weights,float)
    if w.sum() <= 1e-12:
        w=np.full(3,1/3)
    else:
        w=w/w.sum()
    weighted=norm*w
    ideal=weighted.min(0); anti=weighted.max(0)
    ds=np.linalg.norm(weighted-ideal,axis=1); da=np.linalg.norm(weighted-anti,axis=1)
    score=da/(ds+da+1e-12)
    return population[int(np.argmax(score))]
