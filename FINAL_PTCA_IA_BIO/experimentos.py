"""
experimentos.py — Los cuatro experimentos de la práctica. Cada uno escribe resultados/<nombre>.csv, fila a fila, y se
puede interrumpir y reanudar (se salta lo ya hecho).

    python experimentos.py cribado        diseño: variantes en semillas de EXPLORACIÓN 0–1, 7 datasets   (≈ 25 min)
    python experimentos.py confirmacion   el método final en las particiones de CONFIRMACIÓN 40–44, 12 datasets
                                          (+ X original y estandarizado)                                     (≈ 20 min)
    python experimentos.py ablacion       una decisión cambiada cada vez, semillas 40–41, 12 datasets        (≈ 35 min)
    python experimentos.py comparativa    8 transformaciones alternativas en las particiones 40–44          (≈ 8 min)

Tiempos con 20 procesos; N_JOBS=k para usar k (por defecto, todos los núcleos).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import ptca

RES = os.path.join(ptca.CARPETA, "resultados")
CONFIRMACION, EXPLORACION, ABLACION = [40, 41, 42, 43, 44], [0, 1], [40, 41]
DS_CRIBADO = ["california", "diabetes", "concrete_compressive_strength", "puma8NH", "house_8L", "stock", "kin8nm"]

# Variantes: nombre -> argumentos de ptca.aprender (vacío = método final)
VARIANTES = {
    "P": {},
    # cribado de diseño (exploración)
    "fitness solo KNN": dict(fitness=("KNN",)),
    "fitness minimax": dict(agregacion="minimax"),
    "familia RL": dict(fitness=("RL",)),
    "familia árbol": dict(fitness=("Árbol",)),
    "familia RF": dict(fitness=("RF",)),
    "parada (paciencia 15)": dict(parada=("parada", 15)),
    "parada (paciencia 30)": dict(parada=("parada", 30)),
    "elegir θ con apartado": dict(parada=("seleccion", None)),
    # ablación (confirmación)
    "sin síntesis": dict(componentes=False),
    "sin límite n−1": dict(limite=False),
    "sin forma (λ = 1)": dict(forma=False),
    "búsqueda aleatoria": dict(optimizador="aleatoria"),
    "algoritmo genético": dict(optimizador="ag"),
}
EXPERIMENTOS = {
    "cribado": (DS_CRIBADO, EXPLORACION, ["P", "fitness solo KNN", "fitness minimax", "familia RL", "familia árbol",
                                          "familia RF", "parada (paciencia 15)", "parada (paciencia 30)",
                                          "elegir θ con apartado"]),
    "confirmacion": (ptca.DATASETS, CONFIRMACION, ["P"]),
    "ablacion": (ptca.DATASETS, ABLACION, ["fitness solo KNN", "sin síntesis", "sin límite n−1", "sin forma (λ = 1)",
                                           "búsqueda aleatoria", "algoritmo genético"]),
}


def evaluar(A, B, yt, yv, semilla):
    return {m: float(np.mean(np.abs(mod.fit(A, yt).predict(B) - yv))) for m, mod in ptca.modelos(semilla).items()}


def tarea(dataset, semilla, variante, referencias):
    """Aprende P (variante) con el entrenamiento y mide los 4 modelos en validación. Con referencias, también X
    original y X estandarizado (e y la línea base)."""
    Xt, Xv, yt, yv = ptca.particion(dataset, semilla)
    n = Xt.shape[1]
    t0 = time.time()
    P = ptca.aprender(Xt, yt, semilla, **VARIANTES[variante])
    info = dict(segundos=time.time() - t0, dim=P.dim, n_orig=P.n_orig, n_comp=P.n_comp, tau=P.tau,
                evaluaciones=P.evaluaciones, fitness=P.fitness)
    reps = {variante: (P.transform(Xt), P.transform(Xv))}
    if referencias:
        mu, sd = Xt.mean(0), np.where(Xt.std(0) > 0, Xt.std(0), 1)
        reps.update({"original": (Xt, Xv), "estándar": ((Xt - mu) / sd, (Xv - mu) / sd)})
    filas = []
    for rep, (A, B) in reps.items():
        for m, mae in evaluar(A, B, yt, yv, semilla).items():
            filas.append(dict(dataset=dataset, semilla=semilla, n=n, representacion=rep, modelo=m, mae=mae,
                              **(info if rep == variante else {})))
    return filas


def tarea_comparativa(dataset, semilla):
    import alternativas
    alternativas.compatibilidad()
    Xt, Xv, yt, yv = ptca.particion(dataset, semilla)
    filas = []
    for rep, ajustar in alternativas.representaciones(Xt, Xv, yt, semilla).items():
        t0 = time.time()
        A, B = ajustar()
        seg = time.time() - t0
        for m, mae in evaluar(A, B, yt, yv, semilla).items():
            filas.append(dict(dataset=dataset, semilla=semilla, n=Xt.shape[1], representacion=rep, modelo=m, mae=mae,
                              segundos=seg, dim=A.shape[1]))
    return filas


def ejecutar(nombre, variantes=None):
    os.makedirs(RES, exist_ok=True)
    salida = os.path.join(RES, nombre + ".csv")
    hechas = set()
    if os.path.exists(salida):
        hechas = set(map(tuple, pd.read_csv(salida)[["dataset", "semilla", "representacion"]].drop_duplicates().values))
    if nombre == "comparativa":
        trabajo = [(d, s) for s in CONFIRMACION for d in ptca.DATASETS if (d, s, "MLKR") not in hechas]
        fn = tarea_comparativa
    else:
        dss, sems, vs = EXPERIMENTOS[nombre]
        vs = variantes or vs
        trabajo = [(d, s, v, nombre == "confirmacion" or (nombre == "cribado" and v == "P"))
                   for v in vs for s in sems for d in dss if (d, s, v) not in hechas]
        fn = tarea
    print(nombre, ":", len(trabajo), "tareas", flush=True)
    for i, filas in enumerate(Parallel(n_jobs=int(os.environ.get("N_JOBS", os.cpu_count())),
                                       return_as="generator_unordered")(delayed(fn)(*t) for t in trabajo), 1):
        pd.DataFrame(filas).to_csv(salida, mode="a", header=not os.path.exists(salida), index=False)
        print(i, "/", len(trabajo), flush=True)
    print("TERMINADO", nombre, flush=True)


if __name__ == "__main__":
    ejecutar(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else None)
