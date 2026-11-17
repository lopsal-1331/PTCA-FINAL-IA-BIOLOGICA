"""Tercer componente: puerta aprendida (CMA-ES, fitness basado en árboles) que decide qué componentes PCA se AÑADEN a los
atributos estándar. YJ y pesos no cambian a un árbol (monótonos/escala), así que solo actúan vía la PCA.
Variantes: estándar | estándar+PCA (todas) | puerta | YJ+pesos+puerta. Evaluación: KNN, Árbol, RF, ET (200 árboles)."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, "../exploracion")
import numpy as np, pandas as pd, cma, time, warnings
from scipy.stats import yeojohnson
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import pca_lib as P
warnings.filterwarnings("ignore")
SEMILLAS = range(2); F = "resultados_componente3.csv"

def ajustar(Xt, lam, wf, g):
    n = Xt.shape[1]
    C = np.column_stack([yeojohnson(Xt[:, j], lmbda=lam[j]) for j in range(n)]); mu, sd = C.mean(0), C.std(0); sd[sd == 0] = 1
    Zf = wf * (C - mu) / sd; mz = Zf.mean(0); V = np.linalg.svd(Zf - mz, full_matrices=False)[2].T; sel = g > 0.5
    def v(X):
        Z = wf * (np.column_stack([yeojohnson(X[:, j], lmbda=lam[j]) for j in range(n)]) - mu) / sd
        return np.hstack([Z, ((Z - mz) @ V)[:, sel]])
    return v

def aprender(Xs, ys, semilla, n, con_yj, presupuesto=150):
    esc = np.mean(np.abs(ys - np.median(ys))); plie = list(KFold(3, shuffle=True, random_state=semilla).split(Xs))
    def desemp(t): return (t[:n], t[n:2 * n], t[2 * n:]) if con_yj else (np.ones(n), np.ones(n), t)
    def f(t):
        try:
            Z = ajustar(Xs, *desemp(t))(Xs)
            r = np.mean([np.mean(np.abs(ExtraTreesRegressor(20, random_state=semilla, n_jobs=1).fit(Z[a], ys[a]).predict(Z[b]) - ys[b]))
                         for a, b in plie]) / esc
            return r if np.isfinite(r) else 1e6
        except Exception: return 1e6
    if con_yj:
        lo = np.r_[np.full(n, -2.0), np.zeros(n), np.zeros(n)]; hi = np.r_[np.full(n, 2.0), np.full(n, 3.0), np.ones(n)]
        x0 = np.r_[np.ones(2 * n), np.full(n, 0.75)]
    else: lo, hi, x0 = np.zeros(n), np.ones(n), np.full(n, 0.75)
    g = lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo))
    r = cma.fmin(g, (x0 - lo) / (hi - lo), 0.2, options=dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
    return desemp(lo + np.clip(r[0], 0, 1) * (hi - lo))

def tarea(nombre, s):
    X, y = P.cargar(nombre, "../"); n = X.shape[1]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=s)
    sub = np.random.default_rng(s).choice(len(yt), min(1000, len(yt)), replace=False); Xs, ys = Xt[sub], yt[sub]
    uno = np.ones(n); reps = {}
    reps["estándar"] = (ajustar(Xt, uno, uno, np.zeros(n)), 0)
    reps["estándar+PCA (todas)"] = (ajustar(Xt, uno, uno, np.ones(n)), n)
    lam, wf, g = aprender(Xs, ys, s, n, False); reps["puerta"] = (ajustar(Xt, lam, wf, g), int((g > .5).sum()))
    lam, wf, g = aprender(Xs, ys, s, n, True);  reps["YJ+pesos+puerta"] = (ajustar(Xt, lam, wf, g), int((g > .5).sum()))
    out = []
    for rn, (v, nsel) in reps.items():
        A, B = v(Xt), v(Xv)
        mods = {"KNN": KNeighborsRegressor(5), "Árbol": DecisionTreeRegressor(random_state=s),
                "Random Forest": RandomForestRegressor(100, random_state=s, n_jobs=1), "Extra Trees": ExtraTreesRegressor(100, random_state=s, n_jobs=1)}
        for mn, m in mods.items():
            out.append(dict(dataset=nombre, semilla=s, representacion=rn, modelo=mn, n_atributos=n, n_pca_sel=nsel,
                            MAE=float(np.mean(np.abs(m.fit(A, yt).predict(B) - yv)))))
    return out

if __name__ == "__main__":
    trab = [(d, s) for s in SEMILLAS for d in P.DATASETS]
    for i, r in enumerate(Parallel(n_jobs=12, return_as="generator_unordered")(delayed(tarea)(*t) for t in trab), 1):
        pd.DataFrame(r).to_csv(F, mode="a", header=not os.path.exists(F), index=False); print(i, "/", len(trab), flush=True)
    print("TERMINADO", flush=True)
