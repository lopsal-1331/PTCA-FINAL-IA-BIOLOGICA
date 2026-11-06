"""¿Regla lineal local en vez de KNN? ¿Y optimizar la métrica para esa regla?"""
import numpy as np, time, cma, warnings
from sklearn.model_selection import KFold, train_test_split
import lib, pca_lib as P
warnings.filterwarnings("ignore")

def llr(Za, ya, Zb, k, ridge, trozo=300):
    k = min(k, len(Za)); na = (Za ** 2).sum(1); out = np.empty(len(Zb))
    for i in range(0, len(Zb), trozo):
        zb = Zb[i:i + trozo]
        d2 = (zb ** 2).sum(1)[:, None] + na[None, :] - 2 * zb @ Za.T
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dk = np.maximum(np.take_along_axis(d2, idx, 1), 0)
        u = np.sqrt(dk / (dk.max(1, keepdims=True) + 1e-12)); w = (1 - u ** 3) ** 3 + 1e-3
        Dz = Za[idx] - zb[:, None, :]
        A = np.concatenate([np.ones(Dz.shape[:2] + (1,)), Dz], 2); Aw = A * w[..., None]
        M = np.einsum("qki,qkj->qij", Aw, A)
        reg = ridge * w.sum(1)[:, None, None] * np.eye(A.shape[2])[None]; reg[:, 0, 0] = 0
        b = np.einsum("qki,qk->qi", Aw, ya[idx])
        try: out[i:i + trozo] = np.linalg.solve(M + reg + 1e-9 * np.eye(A.shape[2]), b[..., None])[:, 0, 0]
        except np.linalg.LinAlgError: out[i:i + trozo] = ya[idx].mean(1)
    return out

def cv_llr(Zs, ys, semilla, n):
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Zs)); mejor = (np.inf, None)
    for k in sorted({n + 3, 15, 25, 40, 80}):
        for rg in (0.01, 0.1, 1.0):
            v = np.mean([np.mean(np.abs(llr(Zs[a], ys[a], Zs[b], k, rg) - ys[b])) for a, b in plie])
            if v < mejor[0]: mejor = (v, (k, rg))
    return mejor[1]

def fit_llr(Xs, ys, semilla):
    n = Xs.shape[1]; escala = np.mean(np.abs(ys - np.median(ys)))
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Xs))
    def f(th):
        try:
            k = int(round(np.exp(th[2 * n]))); rg = np.exp(th[2 * n + 1])
            Z = P.ajustar(Xs, th[:2 * n], "base")(Xs)
            r = np.mean([np.mean(np.abs(llr(Z[a], ys[a], Z[b], k, rg) - ys[b])) for a, b in plie]) / escala
            return r if np.isfinite(r) else 1e6
        except Exception: return 1e6
    return f

def cma_cubo(g, x0, lo, hi, semilla, presupuesto):
    h = lambda u: g(lo + np.clip(u, 0, 1) * (hi - lo))
    r = cma.fmin(h, (x0 - lo) / (hi - lo), 0.2, options=dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
    return lo + np.clip(r[0], 0, 1) * (hi - lo)

def mae(p, yv): return float(np.mean(np.abs(p - yv)))

def correr(nombre, semilla, raiz="../", presupuesto=1000):
    X, y = P.cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    Xs, ys = Xt[sub], yt[sub]; filas = []
    def reg(metodo, v, regla, seg=0.0):
        Zt, Zv, Zs = v(Xt), v(Xv), v(Xs)
        if regla == "knn":
            plie = list(KFold(5, shuffle=True, random_state=semilla).split(Zs))
            cv = {k: np.mean([np.mean(np.abs(lib.knn_k(Zs[a], ys[a], Zs[b], k) - ys[b])) for a, b in plie]) for k in range(1, 31)}
            k = min(cv, key=cv.get); from sklearn.neighbors import KNeighborsRegressor
            m = mae(KNeighborsRegressor(k).fit(Zt, yt).predict(Zv), yv); extra = (k, np.nan)
        else:
            k, rg = cv_llr(Zs, ys, semilla, n); m = mae(llr(Zt, yt, Zv, k, rg), yv); extra = (k, rg)
        filas.append(dict(dataset=nombre, semilla=semilla, metodo=metodo, mae=m, k=extra[0], ridge=extra[1], segundos=seg))
    mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1; est = lambda X: (X - mu) / sd
    reg("estándar + KNN", est, "knn"); reg("estándar + LLR", est, "llr")
    # métrica optimizada para KNN
    f, _ = P.fitness(Xs, ys, semilla, "base"); x0, lo, hi = P.estructura(n, "base")
    t0 = time.time(); th = cma_cubo(f, x0, lo, hi, semilla, presupuesto); seg = time.time() - t0
    v = P.ajustar(Xt, th, "base")
    reg("métrica(KNN) + KNN", v, "knn", seg); reg("métrica(KNN) + LLR", v, "llr", seg)
    # métrica optimizada para la regla LLR (con k y ridge aprendidos como parámetros de la búsqueda)
    g = fit_llr(Xs, ys, semilla)
    kmin = n + 3
    x0l = np.r_[x0, np.log(max(kmin, 10)), np.log(0.1)]
    lol = np.r_[lo, np.log(kmin), np.log(1e-3)]; hil = np.r_[hi, np.log(80), np.log(10.0)]
    t0 = time.time(); thl = cma_cubo(g, x0l, lol, hil, semilla, presupuesto); seg = time.time() - t0
    v2 = P.ajustar(Xt, thl[:2 * n], "base"); k2 = int(round(np.exp(thl[2 * n]))); r2 = float(np.exp(thl[2 * n + 1]))
    filas.append(dict(dataset=nombre, semilla=semilla, metodo="métrica(LLR) + LLR (optimizada para la regla)",
                      mae=mae(llr(v2(Xt), yt, v2(Xv), k2, r2), yv), k=k2, ridge=r2, segundos=seg))
    return filas
