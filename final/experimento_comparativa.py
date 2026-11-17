"""
experimento_comparativa.py — P frente a alternativas fuertes de preprocesado y aprendizaje de la representación.

Se usan EXACTAMENTE las particiones del experimento de confirmación (semillas 40–44) y los θ que CMA-ES aprendió en
él (guardados en resultados/final.csv), así que no hace falta repetir la búsqueda. Los modelos M son los mismos
cuatro modelos triviales (sin ajustar hiperparámetros): el enunciado fija M y la transformación debe mejorarlo tal
cual.

Representaciones comparadas (todas aprendidas solo con el entrenamiento):
  original, estándar                      referencias
  YJ máx. verosimilitud                   PowerTransformer de scikit-learn (λ por normalidad + estandarizar)
  cuantiles                               QuantileTransformer a normal
  PCA 95 % / PCA v-CV                     "solo PCA": sustituir X por sus componentes (95 % de varianza, o el %
                                          elegido por validación cruzada de KNN en una rejilla)
  estándar + PCA                          "solo PCA" añadida: X estandarizado + todas sus componentes
  MLKR                                    aprendizaje de métricas para regresión (Weinberger y Tesauro, 2007)
  PG (gplearn)                            construcción de atributos con programación genética: X estandarizado +
                                          n atributos simbólicos (SymbolicTransformer)
  P sin síntesis, P                       nuestro método (bloques 1–2 y 1–3)
Se guarda también el tiempo de ajuste de cada transformación (coste).

Uso:  python experimento_comparativa.py      (≈8 min con 20 núcleos)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
import compat  # antes de metric_learn y gplearn
from metric_learn import MLKR
from gplearn.genetic import SymbolicTransformer
import metodo
from experimento_final import modelos

warnings.filterwarnings("ignore")
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
SALIDA = os.path.join(CARPETA, "comparativa.csv")
N_SUB = 1000            # filas de entrenamiento para MLKR y PG (coste cuadrático / poblacional)


def pca(Zt):
    """Componentes de Zt: devuelve (media, V, varianza explicada acumulada)."""
    m = Zt.mean(0)
    _, S, Vt = np.linalg.svd(Zt - m, full_matrices=False)
    return m, Vt.T, np.cumsum(S ** 2) / np.sum(S ** 2)


def v_por_cv(Zt, yt, semilla):
    """% de varianza de la PCA (sustituyendo X) elegido por VC de KNN sobre una submuestra del entrenamiento."""
    idx = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    Zs, ys = Zt[idx], yt[idx]
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Zs))
    mejor = (np.inf, 1.0)
    for v in np.arange(0.5, 1.0001, 0.05):
        err = []
        for a, b in plie:
            m, V, c = pca(Zs[a])
            k = int(np.searchsorted(c, v - 1e-9) + 1)
            err.append(np.mean(np.abs(metodo.knn_rapido((Zs[a] - m) @ V[:, :k], ys[a], (Zs[b] - m) @ V[:, :k]) - ys[b])))
        if np.mean(err) < mejor[0]:
            mejor = (np.mean(err), v)
    return mejor[1]


def representaciones(Xt, Xv, yt, theta, semilla):
    """Diccionario nombre -> (función que ajusta con Xt y devuelve (A, B)). Se cronometra cada una."""
    def estandar():
        s = StandardScaler().fit(Xt)
        return s.transform(Xt), s.transform(Xv)

    def yj_mv():
        p = PowerTransformer(method="yeo-johnson", standardize=True).fit(Xt)
        return p.transform(Xt), p.transform(Xv)

    def cuantiles():
        q = QuantileTransformer(n_quantiles=min(1000, len(Xt)), output_distribution="normal",
                                random_state=semilla).fit(Xt)
        return q.transform(Xt), q.transform(Xv)

    def pca_v(v):
        def f():
            A, B = estandar()
            m, V, c = pca(A)
            vv = v_por_cv(A, yt, semilla) if v is None else v
            k = int(np.searchsorted(c, vv - 1e-9) + 1)
            return (A - m) @ V[:, :k], (B - m) @ V[:, :k]
        return f

    def est_pca():
        A, B = estandar()
        m, V, _ = pca(A)
        return np.hstack([A, (A - m) @ V]), np.hstack([B, (B - m) @ V])

    def mlkr():
        compat.aplicar()                   # por si la librería se cargó tras el parche
        A, B = estandar()
        idx = np.random.default_rng(semilla).choice(len(yt), min(N_SUB, len(yt)), replace=False)
        try:
            L = MLKR(max_iter=100, random_state=semilla).fit(A[idx], yt[idx])
        except TypeError:                      # fallo esporádico de compatibilidad: reparchear el módulo y reintentar
            import traceback
            import metric_learn._util as mu
            print(traceback.format_exc()[-600:], flush=True)
            mu.check_X_y = compat.V.check_X_y
            mu.check_array = compat.V.check_array
            L = MLKR(max_iter=100, random_state=semilla).fit(A[idx], yt[idx])
        return L.transform(A), L.transform(B)

    def pg():
        compat.aplicar()                   # por si la librería se cargó tras el parche
        A, B = estandar()
        idx = np.random.default_rng(semilla).choice(len(yt), min(N_SUB, len(yt)), replace=False)
        g = SymbolicTransformer(population_size=500, generations=10, n_components=A.shape[1], hall_of_fame=50,
                                function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"),
                                parsimony_coefficient=0.001, random_state=semilla, n_jobs=1).fit(A[idx], yt[idx])
        Ga, Gb = g.transform(A), g.transform(B)
        s = StandardScaler().fit(Ga)                               # mismas escalas que el resto de atributos
        Ga, Gb = np.nan_to_num(s.transform(Ga)), np.nan_to_num(s.transform(Gb))
        return np.hstack([A, Ga]), np.hstack([B, Gb])

    def p_metodo(sintesis):
        def f():
            P = metodo.Transformacion(theta, sintesis=sintesis).fit(Xt)
            return P.transform(Xt), P.transform(Xv)
        return f

    return {"original": lambda: (Xt, Xv), "estándar": estandar, "YJ máx. verosimilitud": yj_mv,
            "cuantiles": cuantiles, "PCA 95 %": pca_v(0.95), "PCA v-CV": pca_v(None), "estándar + PCA": est_pca,
            "MLKR": mlkr, "PG (gplearn)": pg, "P sin síntesis": p_metodo(False), "P": p_metodo(True)}


def tarea(nombre, semilla, theta):
    compat.aplicar()                       # en los procesos paralelos la librería puede cargarse antes que el parche
    X, y = metodo.cargar(nombre)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    filas = []
    for rep, ajustar in representaciones(Xt, Xv, yt, theta, semilla).items():
        t0 = time.time()
        try:
            A, B = ajustar()
        except Exception as e:                                      # una alternativa que falle no tumba la tarea
            print("fallo", nombre, semilla, rep, e, flush=True)
            continue
        seg = time.time() - t0
        for mod, m in modelos(semilla).items():
            mae = float(np.mean(np.abs(m.fit(A, yt).predict(B) - yv)))
            filas.append(dict(dataset=nombre, semilla=semilla, representacion=rep, modelo=mod, mae=mae,
                              segundos_transf=seg, n_atributos_salida=A.shape[1]))
    return filas


if __name__ == "__main__":
    fin = pd.read_csv(os.path.join(CARPETA, "final.csv")).drop_duplicates(["dataset", "semilla"])
    # θ = (λ, w) reconstruido de lo que guardó el experimento final; los "segundos" de la búsqueda también valen
    trabajo = [(r.dataset, int(r.semilla), np.r_[np.array(r.lambdas.split("|"), float), np.array(r.pesos.split("|"), float)])
               for r in fin.itertuples()]
    hechas = set()
    if os.path.exists(SALIDA):
        hechas = set(map(tuple, pd.read_csv(SALIDA)[["dataset", "semilla"]].drop_duplicates().values))
    trabajo = [t for t in trabajo if (t[0], t[1]) not in hechas]
    print(len(trabajo), "tareas", flush=True)
    for i, filas in enumerate(Parallel(n_jobs=int(os.environ.get("N_JOBS", os.cpu_count())), return_as="generator_unordered")(
            delayed(tarea)(*t) for t in trabajo), 1):
        pd.DataFrame(filas).to_csv(SALIDA, mode="a", header=not os.path.exists(SALIDA), index=False)
        print(i, "/", len(trabajo), flush=True)
    print("TERMINADO", flush=True)
