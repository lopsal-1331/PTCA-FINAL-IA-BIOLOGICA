"""
experimento_reduccion.py — Cribado de la reducción aprendida por CMA-ES (metodo_reduccion.py).

Mismas particiones que final/experimento_final.py (12 datasets × semillas 40–44, 70/30) y mismos 4 modelos.
Variantes (nombre: alfa, síntesis):
    τ α=0          umbral aprendido, sin penalización (solo la restricción s < n)
    τ α=0.02       umbral + penalización suave por dimensión
    τ α=0.05       umbral + penalización fuerte
    τ+ρ α=0        umbral + síntesis acotada a n-1 atributos en total
Las referencias (original, estándar, P de 2n) están en final/resultados/final.csv.

Uso: python experimento_reduccion.py            (≈5 min con 20 núcleos)
     VARIANTES="τ α=0,τ+ρ α=0" python experimento_reduccion.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(AQUI), "final"))
import metodo                                                       # noqa: E402
from experimento_final import modelos                               # noqa: E402
import metodo_reduccion as mr                                       # noqa: E402

TODAS = {"τ α=0": (0.0, False), "τ α=0.02": (0.02, False), "τ α=0.05": (0.05, False), "τ+ρ α=0": (0.0, True)}
VARIANTES = os.environ.get("VARIANTES", ",".join(TODAS)).split(",")
SEMILLAS = [40, 41, 42, 43, 44]
SALIDA = os.path.join(AQUI, "resultados", "reduccion.csv")


def tarea(nombre, semilla, variante):
    alfa, sint = TODAS[variante]
    X, y = metodo.cargar(nombre)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    t0 = time.time()
    P = mr.aprender(Xt, yt, semilla, alfa=alfa, sintesis=sint)
    seg = time.time() - t0
    A, B = P.transform(Xt), P.transform(Xv)
    n = X.shape[1]
    comun = dict(dataset=nombre, semilla=semilla, variante=variante, n=n, dim=P.dim, s=len(P.S), m=P.m,
                 tau=P.tau, rho=P.rho, fitness0=P.fitness0, fitness=P.fitness, segundos=seg,
                 eliminados="|".join(map(str, sorted(set(range(n)) - set(P.S)))),
                 pesos="|".join(f"{v:.3f}" for v in P.w))
    return [dict(comun, modelo=mod, mae=float(np.mean(np.abs(m.fit(A, yt).predict(B) - yv))))
            for mod, m in modelos(semilla).items()]


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    hechas = set()
    if os.path.exists(SALIDA):
        hechas = set(map(tuple, pd.read_csv(SALIDA)[["dataset", "semilla", "variante"]].drop_duplicates().values))
    trabajo = [(d, s, v) for v in VARIANTES for s in SEMILLAS for d in metodo.DATASETS if (d, s, v) not in hechas]
    print(len(trabajo), "tareas", flush=True)
    for i, filas in enumerate(Parallel(n_jobs=int(os.environ.get("N_JOBS", os.cpu_count())), return_as="generator_unordered")(
            delayed(tarea)(*t) for t in trabajo), 1):
        pd.DataFrame(filas).to_csv(SALIDA, mode="a", header=not os.path.exists(SALIDA), index=False)
        if i % 20 == 0:
            print(i, "/", len(trabajo), flush=True)
    print("TERMINADO", flush=True)
