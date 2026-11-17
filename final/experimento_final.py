"""
experimento_final.py — Experimento de confirmación del método final (el que se reporta en memoria, paper y póster).

Diseño:
  * 12 datasets × 5 particiones 70/30 con semillas NUEVAS (40–44): ninguna se usó para diseñar el método.
  * Para cada partición, P aprende θ con CMA-ES (1000 evaluaciones) usando solo el entrenamiento.
  * Se evalúan 4 modelos M de familias distintas con 4 representaciones:
        original        X tal cual (el error e del enunciado)
        estándar        (X - μ)/σ, la línea base habitual
        P sin síntesis  bloques 1+2 (forma + relevancia)
        P               bloques 1+2+3: X' completo (el error e' del enunciado)
  * Métrica: MAE en el 30 % de validación. Los resultados se guardan fila a fila (se puede interrumpir).

Uso:  python experimento_final.py            (≈10 min con 20 núcleos)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"          # un hilo por proceso: paralelizamos por tarea, no dentro de NumPy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import metodo

SEMILLAS = [int(s) for s in os.environ.get("SEMILLAS", "40,41,42,43,44").split(",")]
PRESUPUESTO = int(os.environ.get("PRESUPUESTO", 1000))
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "final.csv")


def modelos(semilla):
    """Un modelo de cada familia: lineal, basado en distancias, árbol y conjunto de árboles."""
    return {"Regresión lineal": LinearRegression(),
            "KNN": KNeighborsRegressor(5),
            "Árbol": DecisionTreeRegressor(random_state=semilla),
            "Random Forest": RandomForestRegressor(100, random_state=semilla, n_jobs=1)}


def tarea(nombre, semilla):
    X, y = metodo.cargar(nombre)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)

    t0 = time.time()
    P = metodo.aprender(Xt, yt, semilla, PRESUPUESTO)                 # solo ve el entrenamiento
    segundos = time.time() - t0

    mu, sd = Xt.mean(0), Xt.std(0)
    sd[sd == 0] = 1
    sin_sint = metodo.Transformacion(P.theta, sintesis=False).fit(Xt)
    reps = {"original": (Xt, Xv),
            "estándar": ((Xt - mu) / sd, (Xv - mu) / sd),
            "P sin síntesis": (sin_sint.transform(Xt), sin_sint.transform(Xv)),
            "P": (P.transform(Xt), P.transform(Xv))}

    n = X.shape[1]
    comun = dict(dataset=nombre, semilla=semilla, n_filas=len(y), n_atributos=n, segundos=segundos,
                 fitness0=P.fitness0, fitness=P.fitness,
                 lambdas="|".join(f"{v:.3f}" for v in P.lam), pesos="|".join(f"{v:.3f}" for v in P.w))
    filas = []
    for rep, (A, B) in reps.items():
        for mod, m in modelos(semilla).items():
            mae = float(np.mean(np.abs(m.fit(A, yt).predict(B) - yv)))
            filas.append(dict(comun, representacion=rep, modelo=mod, mae=mae))
    return filas


if __name__ == "__main__":
    hechas = set()
    if os.path.exists(SALIDA):                                         # reanudar si se interrumpió
        hechas = set(map(tuple, pd.read_csv(SALIDA)[["dataset", "semilla"]].drop_duplicates().values))
    trabajo = [(d, s) for s in SEMILLAS for d in metodo.DATASETS if (d, s) not in hechas]
    print(len(trabajo), "tareas", flush=True)
    for i, filas in enumerate(Parallel(n_jobs=int(os.environ.get("N_JOBS", os.cpu_count())), return_as="generator_unordered")(
            delayed(tarea)(*t) for t in trabajo), 1):
        pd.DataFrame(filas).to_csv(SALIDA, mode="a", header=not os.path.exists(SALIDA), index=False)
        print(i, "/", len(trabajo), flush=True)
    print("TERMINADO", flush=True)
