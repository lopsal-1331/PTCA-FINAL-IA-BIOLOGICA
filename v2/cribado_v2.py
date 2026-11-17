"""
cribado_v2.py — Cribado corto de P v2 frente al P actual, con semillas de EXPLORACIÓN (no las 40–44 de confirmación).

Variantes:
    actual           reduccion/metodo_reduccion.py (τ+ρ, fitness KNN, 1000 evaluaciones)
    multi            P v2 (pesos dobles sobre 2n candidatos + τ), fitness KNN+RL+árbol, sin parada
    multi+parada     igual + parada por sobreajuste (20 % apartado, paciencia 15 generaciones)
    knn+parada       pesos dobles + parada, pero fitness solo KNN (aísla el efecto del fitness de varios modelos)
Referencias: original y estándar. Modelos medidos en validación: RL, KNN, árbol y Random Forest.

Uso: python cribado_v2.py     (≈10–15 min con 20 núcleos)
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
sys.path.insert(0, AQUI)
import metodo_v2 as v2                                              # noqa: E402
import metodo                                                       # noqa: E402
import metodo_reduccion as mr                                       # noqa: E402
from experimento_final import modelos                               # noqa: E402

DATASETS = os.environ.get("DATASETS", "california,diabetes,concrete_compressive_strength,puma8NH,house_8L,stock").split(",")
SEMILLAS = [int(s) for s in os.environ.get("SEMILLAS", "0,1").split(",")]
VARIANTES = os.environ.get("VARIANTES", "actual,multi,multi+parada,knn+parada").split(",")
SALIDA = os.path.join(AQUI, "resultados", os.environ.get("SALIDA", "cribado_v2.csv"))


def tarea(nombre, semilla, variante):
    X, y = metodo.cargar(nombre)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    t0 = time.time()
    if variante == "actual":
        P = mr.aprender(Xt, yt, semilla, sintesis=True)
        info = dict(dim=P.dim, n_orig=len(P.S), n_comp=P.m, evaluaciones=1000, fitness=P.fitness, validacion=np.nan)
    else:
        M3 = ("KNN", "RL", "Árbol")
        cfg = {"multi": dict(modelos=M3, parada=False),
               "multi+parada": dict(modelos=M3, parada=True),
               "multi+parada30": dict(modelos=M3, parada=True, paciencia=30),
               "multi+selección": dict(modelos=M3, parada=True, paciencia=10 ** 9),   # no corta: solo elige θ
               "knn+parada": dict(modelos=("KNN",), parada=True),
               "rel-media": dict(modelos=M3, parada=False, agregacion="rel-media"),
               "rel-max": dict(modelos=M3, parada=False, agregacion="rel-max"),
               "rel-mix": dict(modelos=M3, parada=False, agregacion="rel-mix"),
               "fam-KNN": dict(familia="KNN", parada=False), "fam-RL": dict(familia="RL", parada=False),
               "fam-Árbol": dict(familia="Árbol", parada=False), "fam-RF": dict(familia="RF", parada=False)}[variante]
        P = v2.aprender(Xt, yt, semilla, **cfg)
        info = dict(dim=P.dim, n_orig=P.n_orig, n_comp=P.n_comp, evaluaciones=P.evaluaciones, fitness=P.fitness,
                    validacion=P.validacion)
    seg = time.time() - t0
    mu, sd = Xt.mean(0), Xt.std(0)
    sd[sd == 0] = 1
    reps = {variante: (P.transform(Xt), P.transform(Xv))}
    if variante == "actual":
        reps.update({"original": (Xt, Xv), "estándar": ((Xt - mu) / sd, (Xv - mu) / sd)})
    filas = []
    for rep, (A, B) in reps.items():
        for mod, m in modelos(semilla).items():
            mae = float(np.mean(np.abs(m.fit(A, yt).predict(B) - yv)))
            filas.append(dict(dataset=nombre, semilla=semilla, n=n, representacion=rep, modelo=mod, mae=mae,
                              segundos=seg if rep == variante else np.nan, **(info if rep == variante else {})))
    return filas


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    hechas = set()
    if os.path.exists(SALIDA):
        hechas = set(map(tuple, pd.read_csv(SALIDA).query("representacion not in ['original','estándar']")
                         [["dataset", "semilla", "representacion"]].drop_duplicates().values))
    trabajo = [(d, s, v) for v in VARIANTES for s in SEMILLAS for d in DATASETS if (d, s, v) not in hechas]
    print(len(trabajo), "tareas", flush=True)
    for i, filas in enumerate(Parallel(n_jobs=int(os.environ.get("N_JOBS", os.cpu_count())),
                                       return_as="generator_unordered")(delayed(tarea)(*t) for t in trabajo), 1):
        pd.DataFrame(filas).to_csv(SALIDA, mode="a", header=not os.path.exists(SALIDA), index=False)
        print(i, "/", len(trabajo), flush=True)
    print("TERMINADO", flush=True)
