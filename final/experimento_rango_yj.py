"""
experimento_rango_yj.py — ¿Por qué λ ∈ [-2, 2]? Ablación del rango de los exponentes Yeo-Johnson.

Se repite la búsqueda del método con λ ∈ [-R, R] para R ∈ {1, 2, 4, 8} y con R = 0, que significa λ fijo a 1
(sin Yeo-Johnson, solo pesos). Para cada búsqueda se guarda:
  * el MAE de KNN en validación (el modelo para el que se optimiza la métrica),
  * el fitness final (error de validación cruzada en entrenamiento),
  * qué fracción de los λ aprendidos acaba pegada al borde del rango (si el límite "molesta", será alta).
Además se calcula el λ de máxima verosimilitud de cada atributo (lo que daría PowerTransformer de scikit-learn)
como referencia de lo extremos que serían los exponentes si se eligieran buscando normalidad.

Uso:  python experimento_rango_yj.py     (≈10 min con 20 núcleos: 3 particiones, 500 evaluaciones)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import yeojohnson
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import metodo

RANGOS = [0, 1, 2, 4, 8]
SEMILLAS = [40, 41, 42]
PRESUPUESTO = 500
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
SALIDA = os.path.join(CARPETA, "rango_yj.csv")


def tarea(nombre, semilla, R):
    X, y = metodo.cargar(nombre)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    lim = (0.999, 1.001) if R == 0 else (-R, R)                 # R = 0: λ congelado en 1 (identidad)
    P = metodo.aprender(Xt, yt, semilla, PRESUPUESTO, sintesis=False, lim_lambda=lim)
    mae = float(np.mean(np.abs(KNeighborsRegressor(5).fit(P.transform(Xt), yt).predict(P.transform(Xv)) - yv)))
    borde = float(np.mean(np.abs(P.lam) > 0.98 * R)) if R else np.nan
    return dict(dataset=nombre, semilla=semilla, R=R, mae_knn=mae, fitness=P.fitness, frac_borde=borde,
                lam_abs_max=float(np.abs(P.lam).max()), lambdas="|".join(f"{v:.3f}" for v in P.lam))


def lambdas_mle():
    """λ de máxima verosimilitud (scipy) de cada atributo en el dataset completo."""
    filas = []
    for nombre in metodo.DATASETS:
        X, _ = metodo.cargar(nombre)
        filas += [dict(dataset=nombre, atributo=j, lam_mle=yeojohnson(X[:, j])[1]) for j in range(X.shape[1])]
    pd.DataFrame(filas).to_csv(os.path.join(CARPETA, "lambdas_mle.csv"), index=False)


if __name__ == "__main__":
    lambdas_mle()
    trabajo = [(d, s, R) for s in SEMILLAS for d in metodo.DATASETS for R in RANGOS]
    for i, fila in enumerate(Parallel(n_jobs=int(os.environ.get("N_JOBS", os.cpu_count())), return_as="generator_unordered")(
            delayed(tarea)(*t) for t in trabajo), 1):
        pd.DataFrame([fila]).to_csv(SALIDA, mode="a", header=not os.path.exists(SALIDA), index=False)
        print(i, "/", len(trabajo), flush=True)
    print("TERMINADO", flush=True)
