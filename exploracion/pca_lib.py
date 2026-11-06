"""Representación aprendida: YJ + pesos por atributo + PCA con atenuación de componentes (reducción de ruido)."""
import numpy as np, pandas as pd, time, cma, warnings
from scipy.stats import yeojohnson
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, train_test_split
import lib
warnings.filterwarnings("ignore")

DATASETS = ["diabetes","california","concrete_compressive_strength","stock","space_ga","sulfur",
            "kin8nm","bank8FM","puma8NH","house_8L","cpu_small","wine_quality"]
def cargar(nombre, raiz="../"):
    if nombre in lib.RUTAS: return lib.cargar(nombre, raiz)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name=nombre, as_frame=True, parser="auto")
    X = d.data.select_dtypes("number").dropna(axis=1).astype(float); y = d.target.astype(float).values
    X = X.loc[:, X.nunique() > 1].values
    if len(y) > 10000:
        i = np.random.default_rng(0).choice(len(y), 10000, replace=False); X, y = X[i], y[i]
    return X, y

# ---- parametrizaciones. theta = (lam[n], wf[n], resto)
def desempaquetar(theta, n, modo):
    lam, wf = theta[:n], theta[n:2 * n]; r = theta[2 * n:]
    if modo == "base":   wp = None                               # sin PCA
    elif modo == "pca_w": wp = r[:n]                             # un peso por componente
    elif modo == "pca_corte":                                    # atenuación suave por rango: 1/(1+exp((j-c)/s))
        j = np.arange(n); c, s = r[0], r[1]; wp = 1 / (1 + np.exp((j - c) / s)) * 1.0
        wp = wp / max(wp[0], 1e-9)                               # el primer componente conserva peso 1
    return lam, wf, wp

def ajustar(Xt, theta, modo):
    n = Xt.shape[1]; lam, wf, wp = desempaquetar(np.asarray(theta, float), n, modo)
    C = np.column_stack([yeojohnson(Xt[:, j], lmbda=lam[j]) for j in range(n)])
    mu, sd = C.mean(0), C.std(0); sd[sd == 0] = 1
    Zf = wf * (C - mu) / sd
    V = None
    if wp is not None:
        _, _, Vt = np.linalg.svd(Zf - Zf.mean(0), full_matrices=False); V = Vt.T; mz = Zf.mean(0)
    def v(X):
        Z = np.column_stack([yeojohnson(X[:, j], lmbda=lam[j]) for j in range(n)])
        Z = wf * (Z - mu) / sd
        return Z if V is None else ((Z - mz) @ V) * wp
    return v

def estructura(n, modo):
    lo = list(np.full(n, -2.0)) + list(np.zeros(n)); hi = list(np.full(n, 2.0)) + list(np.full(n, 3.0))
    x0 = list(np.ones(2 * n))
    if modo == "pca_w":    lo += [0.0] * n; hi += [3.0] * n; x0 += [1.0] * n
    if modo == "pca_corte": lo += [0.5, 0.2]; hi += [n + 2.0, 4.0]; x0 += [n + 1.0, 0.3]   # x0: casi todos los componentes con peso ≈1
    return np.array(x0), np.array(lo), np.array(hi)

def fitness(Xs, ys, semilla, modo):
    n = Xs.shape[1]; escala = np.mean(np.abs(ys - np.median(ys)))
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Xs)); cuenta = [0]
    def f(theta):
        cuenta[0] += 1
        try:
            Z = ajustar(Xs, theta, modo)(Xs)
            r = np.mean([np.mean(np.abs(lib.knn5(Z[a], ys[a], Z[b]) - ys[b])) for a, b in plie]) / escala
            return r if np.isfinite(r) else 1e6
        except Exception:
            return 1e6
    return f, cuenta

def evaluar(Zt, Zv, yt, yv, Zs=None, ys=None, semilla=0):
    d = {"KNN": float(np.mean(np.abs(KNeighborsRegressor(5).fit(Zt, yt).predict(Zv) - yv))),
         "Regresión Lineal": float(np.mean(np.abs(LinearRegression().fit(Zt, yt).predict(Zv) - yv))),
         "Árbol": float(np.mean(np.abs(DecisionTreeRegressor(random_state=13).fit(Zt, yt).predict(Zv) - yv)))}
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Zs))     # k por CV en entrenamiento
    cv = {k: np.mean([np.mean(np.abs(lib.knn_k(Zs[a], ys[a], Zs[b], k) - ys[b])) for a, b in plie]) for k in range(1, 31)}
    k = min(cv, key=cv.get); d["KNN k-CV"] = float(np.mean(np.abs(KNeighborsRegressor(k).fit(Zt, yt).predict(Zv) - yv))); d["k"] = k
    return d

def correr(nombre, semilla, metodo, raiz="../", presupuesto=1000):
    X, y = cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    Xs, ys = Xt[sub], yt[sub]; t0 = time.time(); fb = np.nan
    if metodo == "crudo":      v = lambda X: X
    elif metodo == "estándar":
        mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1; v = lambda X: (X - mu) / sd
    elif metodo == "estándar+PCA95":
        mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1; Z = (Xt - mu) / sd
        _, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); c = np.cumsum(S ** 2) / np.sum(S ** 2)
        m = int(np.searchsorted(c, 0.95) + 1); Vm = Vt[:m].T; mz = Z.mean(0)
        v = lambda X: ((X - mu) / sd - mz) @ Vm
    else:
        modo = {"CMA-ES (YJ+pesos)": "base", "CMA-ES + PCA (peso por componente)": "pca_w", "CMA-ES + PCA (corte suave)": "pca_corte"}[metodo]
        f, cuenta = fitness(Xs, ys, semilla, modo)
        x0, lo, hi = estructura(n, modo)
        g = lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo))
        r = cma.fmin(g, (x0 - lo) / (hi - lo), 0.2, options=dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
        th = lo + np.clip(r[0], 0, 1) * (hi - lo); fb = r[1]
        v = ajustar(Xt, th, modo)
    seg = time.time() - t0
    d = dict(dataset=nombre, semilla=semilla, metodo=metodo, fitness=fb, segundos=seg, n_atributos=n)
    d.update(evaluar(v(Xt), v(Xv), yt, yv, v(Xs), ys, semilla))
    return d
