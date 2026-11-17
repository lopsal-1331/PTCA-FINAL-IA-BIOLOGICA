"""
experimento_busqueda.py — Variantes de la búsqueda con la configuración FINAL del método.

Todas usan el mismo fitness que P (VC de KNN, submuestra de 2000 filas), el mismo presupuesto (1000 evaluaciones),
las particiones de confirmación (semillas 40–44) y los cuatro modelos triviales. El método completo (CMA-ES, λ∈[-2,2],
síntesis con todas las componentes) ya está en resultados/final.csv y sirve de referencia común.

Bloques (python experimento_busqueda.py <bloque>):
  varianza    CMA-ES aprende además v ∈ [0,5, 1], la fracción de varianza que cubren las componentes añadidas
              en la síntesis (es decir, cuántas añadir). 12 datasets × 5 particiones.
  controles   qué aporta cada pieza: solo pesos (λ=1), solo exponentes (w=1), búsqueda aleatoria con el mismo
              presupuesto, pesos por información mutua (sin búsqueda) e YJ de máx. verosimilitud + pesos por
              información mutua. 12 datasets × 2 particiones.
  rango       λ ∈ [-R, R] con R ∈ {1, 4, 8} (R = 2 es el método; R = 0 es "solo pesos"). 12 × 2.
  algoritmos  otros optimizadores con el mismo θ, fitness y presupuesto: AG propio, DE, PSO, BBO, HS y ABC (mealpy).
              12 × 2.
  inicio      la distribución inicial de CMA-ES: media en un punto aleatorio en vez de θ0 (estandarización),
              σ0 = 0,1 o 0,4 en vez de 0,2, y el doble de población que el valor por defecto 4 + ⌊3 ln D⌋. 12 × 2.
Cada fila guarda el MAE de los cuatro modelos con X' (síntesis incluida), el fitness y el tiempo de búsqueda.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import time
import importlib
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PowerTransformer
import metodo
from experimento_final import modelos

warnings.filterwarnings("ignore")
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
SALIDA = os.path.join(CARPETA, "busqueda.csv")
PRESUPUESTO = 1000

BLOQUES = {
    "varianza": ([40, 41, 42, 43, 44], ["CMA-ES + v"]),
    "controles": ([40, 41], ["solo pesos", "solo exponentes", "búsqueda aleatoria", "pesos MI",
                                 "YJ máx. verosimilitud + pesos MI"]),
    "rango": ([40, 41], ["R=1", "R=4", "R=8"]),
    "algoritmos": ([40, 41], ["AG (propio)", "DE", "PSO", "BBO", "HS", "ABC"]),
    "inicio": ([40, 41], ["inicio aleatorio", "sigma0=0.1", "sigma0=0.4", "población x2"]),
}
MEALPY = {"DE": "DE", "PSO": "PSO", "BBO": "BBO", "HS": "HS", "ABC": "ABC"}


# ------------------------------------------------------------------ optimizadores (todos minimizan f en [lo, hi])
def cma_es(f, lo, hi, th0, semilla):
    return metodo.buscar_cma(f, lo, hi, th0, PRESUPUESTO, semilla)[0]


def cma_config(f, lo, hi, x0, semilla, sigma0=0.2, popsize=None):
    """CMA-ES con media inicial, σ0 y tamaño de población configurables (en el cubo [0,1]^D, como metodo.buscar_cma)."""
    import cma
    opciones = dict(seed=semilla + 1, bounds=[0, 1], maxfevals=PRESUPUESTO, verbose=-9)
    if popsize:
        opciones["popsize"] = popsize
    r = cma.fmin(lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo)), (x0 - lo) / (hi - lo), sigma0, options=opciones)
    return lo + np.clip(r[0], 0, 1) * (hi - lo)


def aleatoria(f, lo, hi, th0, semilla):
    """El mejor de θ0 y 999 puntos uniformes en los límites."""
    rng = np.random.default_rng(semilla)
    mejor, th = f(th0), th0
    for _ in range(PRESUPUESTO - 1):
        c = rng.uniform(lo, hi)
        v = f(c)
        if v < mejor:
            mejor, th = v, c
    return th


def ag_propio(f, lo, hi, th0, semilla, N=30):
    """AG con torneo de 3, cruce BLX-α (α=0,3), mutación gaussiana (p=1/D) y elitismo de 1. θ0 en la población."""
    rng = np.random.default_rng(semilla)
    D = len(lo)
    P = np.vstack([th0, rng.uniform(lo, hi, (N - 1, D))])
    F = np.array([f(p) for p in P])
    usadas = N
    while usadas < PRESUPUESTO:
        hijos = []
        while len(hijos) < N - 1 and usadas + len(hijos) < PRESUPUESTO:
            i, j = [min(rng.integers(0, N, 3), key=lambda k: F[k]) for _ in range(2)]
            c = P[i] + rng.uniform(-0.3, 1.3, D) * (P[j] - P[i])
            mut = rng.random(D) < 1.0 / D
            c = np.where(mut, c + rng.normal(0, 0.1, D) * (hi - lo), c)
            hijos.append(np.clip(c, lo, hi))
        Fh = np.array([f(h) for h in hijos])
        usadas += len(hijos)
        e = np.argmin(F)
        P, F = np.vstack([P[e], hijos]), np.r_[F[e], Fh]
        N = len(P)
    return P[np.argmin(F)]


def con_mealpy(nombre):
    """Envuelve un optimizador de mealpy con tope DURO de evaluaciones y θ0 en la población inicial."""
    def opt(f, lo, hi, th0, semilla, pop=30):
        from mealpy import FloatVar, Optimizer
        mod = getattr(importlib.import_module("mealpy"), nombre)         # p. ej. mealpy.DE, mealpy.HS
        clase = getattr(mod, "Original" + nombre, None) or next(
            getattr(mod, c) for c in dir(mod) if isinstance(getattr(mod, c), type)
            and issubclass(getattr(mod, c), Optimizer) and getattr(mod, c) is not Optimizer)
        rng = np.random.default_rng(semilla)
        ini = [th0] + [rng.uniform(lo, hi) for _ in range(pop - 1)]
        estado = {"n": 0, "v": np.inf, "x": th0}

        class Tope(Exception):
            pass

        def obj(x):
            if estado["n"] >= PRESUPUESTO:
                raise Tope()
            estado["n"] += 1
            v = f(np.asarray(x, float))
            if v < estado["v"]:
                estado["v"], estado["x"] = v, np.array(x, float)
            return v
        prob = {"obj_func": obj, "bounds": FloatVar(lb=lo, ub=hi), "minmax": "min", "log_to": None}
        try:
            m = clase(epoch=100000, pop_size=pop)
        except TypeError:
            m = clase(epoch=100000)
        try:
            try:
                m.solve(prob, termination={"max_fe": PRESUPUESTO}, seed=semilla, starting_solutions=ini)
            except TypeError:
                m.solve(prob, termination={"max_fe": PRESUPUESTO}, seed=semilla)
        except Tope:
            pass
        return estado["x"]
    return opt


# ------------------------------------------------------------------ una búsqueda
def tarea(variante, nombre, semilla):
    X, y = metodo.cargar(nombre)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    idx = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    Xs, ys = Xt[idx], yt[idx]
    lo, hi, th0 = metodo.limites(n)
    f = metodo.hacer_fitness(Xs, ys, semilla)
    t0 = time.time()

    if variante == "CMA-ES + v":                     # θ = (λ, w, v); la síntesis parcial sí cambia KNN
        lo, hi, th0 = np.r_[lo, 0.5], np.r_[hi, 1.0], np.r_[th0, 1.0]
        f = metodo.hacer_fitness(Xs, ys, semilla, sintesis=True)
        th = cma_es(f, lo, hi, th0, semilla)
    elif variante == "solo pesos":
        lo[:n], hi[:n] = 0.999, 1.001
        th = cma_es(f, lo, hi, th0, semilla)
    elif variante == "solo exponentes":
        lo[n:], hi[n:] = 0.999, 1.001
        th = cma_es(f, lo, hi, th0, semilla)
    elif variante.startswith("R="):
        R = float(variante[2:])
        lo[:n], hi[:n] = -R, R
        th = cma_es(f, lo, hi, th0, semilla)
    elif variante == "inicio aleatorio":            # media inicial uniforme en los límites, no θ0
        th = cma_config(f, lo, hi, np.random.default_rng(semilla + 100).uniform(lo, hi), semilla)
    elif variante.startswith("sigma0="):
        th = cma_config(f, lo, hi, th0, semilla, sigma0=float(variante.split("=")[1]))
    elif variante == "población x2":
        th = cma_config(f, lo, hi, th0, semilla, popsize=2 * (4 + int(3 * np.log(len(lo)))))
    elif variante == "búsqueda aleatoria":
        th = aleatoria(f, lo, hi, th0, semilla)
    elif variante == "AG (propio)":
        th = ag_propio(f, lo, hi, th0, semilla)
    elif variante in MEALPY:
        th = con_mealpy(variante)(f, lo, hi, th0, semilla)
    elif variante in ("pesos MI", "YJ máx. verosimilitud + pesos MI"):
        # sin búsqueda: pesos proporcionales a la información mutua con y (el mayor vale 3), λ = 1 o el de máx. verosim.
        lam = np.ones(n)
        if variante != "pesos MI":
            lam = np.clip(PowerTransformer(method="yeo-johnson").fit(Xt).lambdas_, -2, 2)
        Z = metodo.Transformacion(np.r_[lam, np.ones(n)], sintesis=False).fit(Xt).transform(Xt)
        mi = mutual_info_regression(Z, yt, random_state=semilla)
        th = np.r_[lam, 3 * mi / max(mi.max(), 1e-12)]
    else:
        raise ValueError(variante)
    segundos = time.time() - t0

    P = metodo.Transformacion(th).fit(Xt)
    A, B = P.transform(Xt), P.transform(Xv)
    fila = dict(variante=variante, dataset=nombre, semilla=semilla, fitness=f(th), segundos=segundos,
                lambdas="|".join(f"{v:.3f}" for v in th[:n]), pesos="|".join(f"{v:.3f}" for v in th[n:2 * n]),
                v=float(th[2 * n]) if len(th) > 2 * n else 1.0, n_sintesis=P.m)
    for mod, m in modelos(semilla).items():
        fila["mae " + mod] = float(np.mean(np.abs(m.fit(A, yt).predict(B) - yv)))
    return fila


if __name__ == "__main__":
    bloque = sys.argv[1]
    semillas, variantes = BLOQUES[bloque]
    hechas = set()
    if os.path.exists(SALIDA):
        hechas = set(map(tuple, pd.read_csv(SALIDA)[["variante", "dataset", "semilla"]].values))
    trabajo = [(v, d, s) for s in semillas for d in metodo.DATASETS for v in variantes if (v, d, s) not in hechas]
    print(bloque, len(trabajo), "tareas", flush=True)
    for i, fila in enumerate(Parallel(n_jobs=int(os.environ.get("N_JOBS", os.cpu_count())), return_as="generator_unordered")(
            delayed(tarea)(*t) for t in trabajo), 1):
        pd.DataFrame([fila]).to_csv(SALIDA, mode="a", header=not os.path.exists(SALIDA), index=False)
        print(i, "/", len(trabajo), flush=True)
    print("TERMINADO", bloque, flush=True)
