"""¿La PCA mejora a modelos basados en orden (árboles/RF) aunque a KNN le aporte poco?
Representaciones: estándar | PCA completa (solo rotación) | PCA95 (reducción) | estándar+PCA (concatenadas).
Modelos: KNN(5), Árbol, Random Forest, Extra Trees. Métrica: MAE en test (30 % hold-out), 10 semillas, 12 datasets."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, "../exploracion")
import numpy as np, pandas as pd, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import pca_lib
warnings.filterwarnings("ignore")
SEMILLAS = range(10); F = "resultados.csv"

def representaciones(Xt, Xv):
    mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1
    Zt, Zv = (Xt - mu) / sd, (Xv - mu) / sd
    mz = Zt.mean(0); _, S, Vt = np.linalg.svd(Zt - mz, full_matrices=False); V = Vt.T
    c = np.cumsum(S ** 2) / np.sum(S ** 2); m = int(np.searchsorted(c, 0.95) + 1)
    Pt, Pv = (Zt - mz) @ V, (Zv - mz) @ V
    return {"estándar": (Zt, Zv), "PCA completa": (Pt, Pv), "PCA95": (Pt[:, :m], Pv[:, :m]),
            "estándar+PCA": (np.hstack([Zt, Pt]), np.hstack([Zv, Pv]))}, m

def modelos(s):
    return {"KNN": KNeighborsRegressor(5), "Árbol": DecisionTreeRegressor(random_state=s),
            "Random Forest": RandomForestRegressor(200, random_state=s, n_jobs=1),
            "Extra Trees": ExtraTreesRegressor(200, random_state=s, n_jobs=1)}

def tarea(nombre, s):
    X, y = pca_lib.cargar(nombre, "../")
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=s)
    reps, m = representaciones(Xt, Xv); out = []
    for rn, (A, B) in reps.items():
        for mn, mod in modelos(s).items():
            out.append(dict(dataset=nombre, semilla=s, representacion=rn, modelo=mn, n_atributos=X.shape[1], m95=m,
                            MAE=float(np.mean(np.abs(mod.fit(A, yt).predict(B) - yv)))))
    return out

if __name__ == "__main__":
    trab = [(d, s) for s in SEMILLAS for d in pca_lib.DATASETS]; res = []
    for i, r in enumerate(Parallel(n_jobs=18, return_as="generator_unordered")(delayed(tarea)(*t) for t in trab), 1):
        res += r; print(i, "/", len(trab), flush=True)
    pd.DataFrame(res).to_csv(F, index=False); print("TERMINADO")
