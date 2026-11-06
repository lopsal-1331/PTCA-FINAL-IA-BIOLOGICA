"""Métrica aprendida para KNN-regresión con estrategias evolutivas (experimentos)."""
import numpy as np, pandas as pd, cma, time, warnings
from scipy.stats import yeojohnson
from scipy.optimize import minimize
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import mutual_info_regression
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- datos
def cargar(nombre, raiz="../"):
    if nombre in ("diabetes", "california"):
        f, t = {"diabetes": ("DATASETS/Diabetes.csv", "target"),
                "california": ("DATASETS/California.csv", "MedHouseVal")}[nombre]
        df = pd.read_csv(raiz + f)
        return df.drop(columns=t).values.astype(float), df[t].values.astype(float)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name=nombre, as_frame=True, parser="auto")
    X = d.data.select_dtypes("number").dropna(axis=1).astype(float)
    y = d.target.astype(float).values
    X = X.loc[:, X.nunique() > 1].values
    if len(y) > 10000:
        i = np.random.default_rng(0).choice(len(y), 10000, replace=False)
        X, y = X[i], y[i]
    return X, y

# ---------------------------------------------------------- transformación
def ajustar(Xt, theta):
    """theta = (lambdas, pesos). Devuelve función que transforma X."""
    n = Xt.shape[1]; lam, w = theta[:n], theta[n:]
    cols = [yeojohnson(Xt[:, j], lmbda=lam[j]) for j in range(n)]
    mu = np.array([c.mean() for c in cols]); sd = np.array([c.std() or 1.0 for c in cols])
    def v(X):
        Z = np.column_stack([yeojohnson(X[:, j], lmbda=lam[j]) for j in range(n)])
        return w * (Z - mu) / sd
    return v

def estandar(Xt):
    mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1
    return lambda X: (X - mu) / sd

# ------------------------------------------------------------------ fitness
K = 5
def knn_rapido(Za, ya, Zb, k):
    """KNN-regresión por fuerza bruta (misma predicción que sklearn salvo empates)."""
    d = (Zb ** 2).sum(1)[:, None] + (Za ** 2).sum(1)[None, :] - 2 * Zb @ Za.T
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    return ya[idx].mean(1)
def hacer_fitness(Xs, ys, semilla, k_variable=False, n_pl=5):
    escala = np.mean(np.abs(ys - np.median(ys)))
    plie = list(KFold(n_pl, shuffle=True, random_state=semilla).split(Xs))
    n = Xs.shape[1]
    def f(theta):
        theta = np.asarray(theta, float)
        k = K
        if k_variable:
            k = int(np.clip(round(np.exp(theta[-1])), 1, 50)); theta = theta[:-1]
        try:
            Z = ajustar(Xs, theta)(Xs)   # media y desviación con X de entrenamiento (sin usar y)
            tot = 0.0
            for a, b in plie:
                tot += np.mean(np.abs(knn_rapido(Z[a], ys[a], Z[b], k) - ys[b]))
            r = tot / n_pl / escala
            return r if np.isfinite(r) else 1e6
        except Exception:
            return 1e6
    return f

# ------------------------------------------- CMA-ES (librería de referencia)
def cma_buscar(f, x0, lo, hi, semilla, presupuesto=2020, sigma0=0.2, ipop=False):
    """Cubo normalizado [0,1]^D; mismo presupuesto de evaluaciones que el notebook final."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    g = lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo))
    u0 = (np.asarray(x0, float) - lo) / (hi - lo)
    opts = dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9,
                tolfun=1e-9, tolx=1e-9)
    if ipop:
        res = cma.fmin(g, u0, sigma0, options=opts, restarts=6, incpopsize=2, bipop=False)
    else:
        res = cma.fmin(g, u0, sigma0, options=opts)
    return lo + np.clip(res[0], 0, 1) * (hi - lo), res[1]

# ---------------------------------- control gradiente: regresión por kernel
def metrica_kernel(Xs, ys, n_max=700, semilla=0):
    """Métrica diagonal por descenso de gradiente en la regresión por kernel suave
    (variante de Weinberger y Tesauro 2007). Devuelve pesos a_f >= 0 sobre datos estandarizados."""
    rng = np.random.default_rng(semilla)
    if len(ys) > n_max:
        i = rng.choice(len(ys), n_max, replace=False); Xs, ys = Xs[i], ys[i]
    ys = (ys - ys.mean()) / (ys.std() or 1)
    D = (Xs[:, None, :] - Xs[None, :, :]) ** 2          # n,n,f
    n, F = len(ys), Xs.shape[1]
    def obj(s):
        a = np.exp(s)
        d2 = D @ a; np.fill_diagonal(d2, np.inf)
        d2 = d2 - d2.min(1, keepdims=True)
        P = np.exp(-d2); P /= P.sum(1, keepdims=True)
        pred = P @ ys; err = pred - ys
        L = np.mean(err ** 2)
        Dbar = np.einsum("ij,ijf->if", P, D)
        PyD = np.einsum("ij,j,ijf->if", P, ys, D)
        dpred = -(PyD - pred[:, None] * Dbar)           # d pred_i / d a_f
        g = (2 / n) * (err[:, None] * dpred).sum(0) * a  # cadena con a = exp(s)
        return L, g
    s0 = np.full(F, np.log(1.0 / F))
    r = minimize(obj, s0, jac=True, method="L-BFGS-B", bounds=[(-8, 3)] * F,
                 options=dict(maxiter=100))
    return np.exp(r.x)

# ------------------------------------------------------------- evaluación
def mae_knn(Zt, Zv, yt, yv, k=K):
    return float(np.mean(np.abs(KNeighborsRegressor(k).fit(Zt, yt).predict(Zv) - yv)))

def un_trabajo(nombre, semilla, raiz="../", presupuesto=2020):
    X, y = cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    Xs, ys = Xt[sub], yt[sub]
    filas = []
    def reg(metodo, v, k=K, seg=0.0, fit=np.nan, extra=None):
        d = dict(dataset=nombre, semilla=semilla, metodo=metodo, segundos=seg, fitness=fit)
        for kk in (3, 5, 10, 20):
            d[f"mae_k{kk}"] = mae_knn(v(Xt), v(Xv), yt, yv, kk)
        d["mae"] = mae_knn(v(Xt), v(Xv), yt, yv, k)
        d.update(extra or {}); filas.append(d)

    I = np.concatenate([np.ones(n), np.ones(n)])
    lo = np.r_[np.full(n, -2.0), np.zeros(n)]; hi = np.r_[np.full(n, 2.0), np.full(n, 3.0)]
    # --- líneas base y heurísticas sin búsqueda
    reg("crudo", lambda X: X)
    reg("estándar", estandar(Xt))
    reg("yj+std", ajustar(Xt, np.r_[np.ones(n), np.ones(n)]))
    mi = mutual_info_regression(estandar(Xt)(Xs), ys, random_state=semilla); mi = mi / (mi.max() or 1)
    reg("pesos MI", lambda X, e=estandar(Xt): e(X) * mi)
    # --- control por gradiente (regresión por kernel)
    t0 = time.time(); a = metrica_kernel(estandar(Xt)(Xs), ys, semilla=semilla); seg = time.time() - t0
    reg("gradiente (kernel)", lambda X, e=estandar(Xt): e(X) * np.sqrt(a), seg=seg)
    # --- CMA-ES con el fitness del notebook final es LR+KNN; aquí: solo KNN
    f = hacer_fitness(Xs, ys, semilla)
    x0 = np.r_[np.ones(n), np.ones(n)]
    t0 = time.time(); th, fb = cma_buscar(f, x0, lo, hi, semilla, presupuesto); seg = time.time() - t0
    reg("CMA-ES (KNN)", ajustar(Xt, th), seg=seg, fit=fb)
    # --- IPOP
    t0 = time.time(); th2, fb2 = cma_buscar(f, x0, lo, hi, semilla, presupuesto, ipop=True); seg = time.time() - t0
    reg("IPOP-CMA-ES (KNN)", ajustar(Xt, th2), seg=seg, fit=fb2)
    # --- híbrido memético: arranca de la métrica por gradiente
    w0 = np.clip(np.sqrt(a) / np.sqrt(a).max() * 1.5, 0.05, 3.0)
    t0 = time.time(); th3, fb3 = cma_buscar(f, np.r_[np.ones(n), w0], lo, hi, semilla, presupuesto, sigma0=0.1)
    seg = time.time() - t0
    reg("gradiente→CMA-ES", ajustar(Xt, th3), seg=seg, fit=fb3)
    # --- k aprendido conjuntamente
    fk = hacer_fitness(Xs, ys, semilla, k_variable=True)
    lok, hik = np.r_[lo, 0.0], np.r_[hi, np.log(50)]
    t0 = time.time(); th4, fb4 = cma_buscar(fk, np.r_[x0, np.log(K)], lok, hik, semilla, presupuesto)
    seg = time.time() - t0
    k4 = int(np.clip(round(np.exp(th4[-1])), 1, 50))
    reg("CMA-ES (KNN, k aprendido)", ajustar(Xt, th4[:-1]), k=k4, seg=seg, fit=fb4, extra=dict(k_aprendido=k4))
    return filas
