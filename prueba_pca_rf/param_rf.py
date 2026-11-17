"""Cribado rápido: ¿qué parámetro/atributo extra, construido sobre el espacio aprendido (CMA-ES YJ+pesos, fitness KNN),
reduce el MAE de Random Forest? RF es invariante a YJ y a escalas, así que solo cambian cosas que alteren el conjunto de atributos."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, "../exploracion")
import numpy as np, pandas as pd, cma, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pca_lib as P
warnings.filterwarnings("ignore")
SEMILLAS = range(int(os.environ.get("NSEM", 2))); PRESUP = int(os.environ.get("PRESUP", 300)); F = os.environ.get("F", "resultados_param_rf.csv")

def oof(modelo, A, y, B, s):
    """Predicción fuera de fold en train (A) y ajustada en todo train para test (B)."""
    o = np.zeros(len(y))
    for a, b in KFold(5, shuffle=True, random_state=s).split(A): o[b] = modelo().fit(A[a], y[a]).predict(A[b])
    return o[:, None], modelo().fit(A, y).predict(B)[:, None]

def tarea(nombre, s):
    X, y = P.cargar(nombre, "../"); n = X.shape[1]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=s)
    sub = np.random.default_rng(s).choice(len(yt), min(2000, len(yt)), replace=False)
    f, _ = P.fitness(Xt[sub], yt[sub], s, "base"); x0, lo, hi = P.estructura(n, "base")
    r = cma.fmin(lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo)), (x0 - lo) / (hi - lo), 0.2,
                 options=dict(seed=s + 1, bounds=[0, 1], maxfevals=PRESUP, verbose=-9))
    th = lo + np.clip(r[0], 0, 1) * (hi - lo); wf = th[n:2 * n]; v = P.ajustar(Xt, th, "base")
    Lt, Lv = v(Xt), v(Xv)                                   # espacio aprendido (para RF ≡ estándar salvo pesos 0)
    mz = Lt.mean(0); V = np.linalg.svd(Lt - mz, full_matrices=False)[2].T; Pt, Pv = (Lt - mz) @ V, (Lv - mz) @ V
    Kt, Kv = oof(lambda: KNeighborsRegressor(10), Lt, yt, Lv, s)
    Rt, Rv = oof(lambda: Ridge(1.0), Lt, yt, Lv, s)
    act = wf / wf.max() >= 0.1
    reps = {"estándar": (Xt, Xv), "+PCA aprendida": (np.hstack([Lt, Pt]), np.hstack([Lv, Pv])),
            "selección wf≥0.1·max": (Lt[:, act], Lv[:, act]),
            "+KNN oof": (np.hstack([Lt, Kt]), np.hstack([Lv, Kv])), "+lineal oof": (np.hstack([Lt, Rt]), np.hstack([Lv, Rv])),
            "+KNN oof +lineal oof": (np.hstack([Lt, Kt, Rt]), np.hstack([Lv, Kv, Rv])),
            "+PCA +KNN oof": (np.hstack([Lt, Pt, Kt]), np.hstack([Lv, Pv, Kv]))}
    # parámetro discreto "bloque extra": se elige por CV interna (3 pliegues, RF 50 árboles, ≤2000 filas de train)
    cand = ["estándar", "+PCA aprendida", "+KNN oof", "+KNN oof +lineal oof", "+PCA +KNN oof"]
    idx = np.random.default_rng(s + 7).choice(len(yt), min(2000, len(yt)), replace=False)
    def cv(k):
        A = reps[k][0][idx]; yy = yt[idx]
        return np.mean([np.mean(np.abs(RandomForestRegressor(50, random_state=s, n_jobs=1).fit(A[a], yy[a]).predict(A[b]) - yy[b]))
                        for a, b in KFold(3, shuffle=True, random_state=s).split(A)])
    elegido = min(cand, key=cv); reps["selección por CV"] = reps[elegido]
    out = [dict(elegido=elegido, dataset=nombre, semilla=s, representacion=k, n_activos=int(act.sum()), n=n,
                MAE=float(np.mean(np.abs(RandomForestRegressor(100, random_state=s, n_jobs=1).fit(A, yt).predict(B) - yv))))
           for k, (A, B) in reps.items()]
    out.append(dict(dataset=nombre, semilla=s, representacion="KNN (referencia)", n_activos=n, n=n,
                    MAE=float(np.mean(np.abs(KNeighborsRegressor(5).fit(Lt, yt).predict(Lv) - yv)))))
    return out

if __name__ == "__main__":
    trab = [(d, s) for s in SEMILLAS for d in P.DATASETS]
    for i, r in enumerate(Parallel(n_jobs=20, return_as="generator_unordered")(delayed(tarea)(*t) for t in trab), 1):
        pd.DataFrame(r).to_csv(F, mode="a", header=not os.path.exists(F), index=False); print(i, "/", len(trab), flush=True)
    print("TERMINADO", flush=True)
