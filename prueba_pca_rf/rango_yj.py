"""Justificación del rango de los exponentes Yeo-Johnson: CMA-ES (YJ+pesos, fitness KNN) con λ ∈ [-R, R].
R=0 → λ fijo a 1 (solo pesos): ablación para ver si YJ aporta algo. Además guarda los λ aprendidos (¿pegan en el borde?)
y los λ de máxima verosimilitud (scipy) de cada atributo como referencia independiente del optimizador."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, "../exploracion")
import numpy as np, pandas as pd, cma, time, warnings
from scipy.stats import yeojohnson
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import pca_lib as P
warnings.filterwarnings("ignore")
RANGOS = [0, 1, 2, 4, 8]; SEMILLAS = range(1); F = "resultados_rango_yj.csv"

def correr(nombre, semilla, R, presupuesto=300):
    X, y = P.cargar(nombre, "../")
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla); n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False); Xs, ys = Xt[sub], yt[sub]
    f, _ = P.fitness(Xs, ys, semilla, "base")
    r = 1e-3 if R == 0 else R; c = 1.0 if R == 0 else 0.0           # R=0: λ ∈ [0.999, 1.001]
    lo = np.r_[np.full(n, c - r), np.zeros(n)]; hi = np.r_[np.full(n, c + r), np.full(n, 3.0)]; x0 = np.ones(2 * n)
    g = lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo)); t0 = time.time()
    res = cma.fmin(g, (x0 - lo) / (hi - lo), 0.2, options=dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
    th = lo + np.clip(res[0], 0, 1) * (hi - lo); lam = th[:n]; v = P.ajustar(Xt, th, "base")
    d = dict(dataset=nombre, semilla=semilla, R=R, fitness=res[1], segundos=time.time() - t0, n_atributos=n,
             frac_borde=float(np.mean(np.abs(lam - c) > 0.98 * r)) if R else np.nan,
             lam_abs_max=float(np.abs(lam).max()), lam="|".join(f"{l:.3f}" for l in lam))
    d.update(P.evaluar(v(Xt), v(Xv), yt, yv, v(Xs), ys, semilla)); return d

def mle():
    filas = []
    for nombre in P.DATASETS:
        X, _ = P.cargar(nombre, "../")
        for j in range(X.shape[1]):
            filas.append(dict(dataset=nombre, atributo=j, lam_mle=yeojohnson(X[:, j])[1], asimetria=float(pd.Series(X[:, j]).skew())))
    pd.DataFrame(filas).to_csv("lambdas_mle.csv", index=False)

if __name__ == "__main__":
    mle(); trab = [(d, s, R) for s in SEMILLAS for d in P.DATASETS for R in RANGOS]; n = 0
    for r in Parallel(n_jobs=10, return_as="generator_unordered")(delayed(correr)(*t) for t in trab):
        pd.DataFrame([r]).to_csv(F, mode="a", header=not os.path.exists(F), index=False); n += 1; print(n, "/", len(trab), flush=True)
    print("TERMINADO", flush=True)
