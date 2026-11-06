"""Comparación con el estándar estadístico: regresión lineal local con núcleo gaussiano
y ancho de banda por variable (tipo Hall-Li-Racine) frente a CMA-ES y frente al gradiente."""
import numpy as np, time, warnings
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
import knn_lib as L
warnings.filterwarnings("ignore")
KG = 150          # vecinos candidatos (el núcleo gaussiano pesa casi cero más allá)
RIDGE = 1e-2

def llr_gauss(Za, ya, Zb, k=KG, ridge=RIDGE, trozo=300):
    """Lineal local con núcleo gaussiano exp(-d²/2) sobre datos ya escalados por 1/h."""
    k = min(k, len(Za)); na = (Za ** 2).sum(1); out = np.empty(len(Zb))
    for i in range(0, len(Zb), trozo):
        zb = Zb[i:i + trozo]
        d2 = (zb ** 2).sum(1)[:, None] + na[None, :] - 2 * zb @ Za.T
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dk = np.maximum(np.take_along_axis(d2, idx, 1), 0)
        w = np.exp(-0.5 * dk) + 1e-12
        Dz = Za[idx] - zb[:, None, :]
        A = np.concatenate([np.ones(Dz.shape[:2] + (1,)), Dz], 2)
        Aw = A * w[..., None]
        M = np.einsum("qki,qkj->qij", Aw, A)
        reg = ridge * w.sum(1)[:, None, None] * np.eye(A.shape[2])[None]; reg[:, 0, 0] = 0
        b = np.einsum("qki,qk->qi", Aw, ya[idx])
        try:
            out[i:i + trozo] = np.linalg.solve(M + reg + 1e-9 * np.eye(A.shape[2]), b[..., None])[:, 0, 0]
        except np.linalg.LinAlgError:
            out[i:i + trozo] = ya[idx].mean(1)
    return out

def fit_llrg(Xs, ys, semilla, con_yj):
    n = Xs.shape[1]; escala = np.mean(np.abs(ys - np.median(ys)))
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Xs))
    def f(u):
        u = np.asarray(u, float)
        theta = np.r_[u[:n], np.exp(u[n:])] if con_yj else np.r_[np.ones(n), np.exp(u)]
        try:
            Z = L.ajustar(Xs, theta)(Xs); tot = 0.0
            for a, b in plie:
                tot += np.mean(np.abs(llr_gauss(Z[a], ys[a], Z[b]) - ys[b]))
            r = tot / 5 / escala
            return r if np.isfinite(r) else 1e6
        except Exception:
            return 1e6
    return f

def trabajo_est(nombre, semilla, metodo, raiz="../", presupuesto=2020):
    X, y = L.cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    Xs, ys = Xt[sub], yt[sub]
    lw = np.log(0.01), np.log(3.0)
    t0 = time.time()
    if metodo == "LLR-gauss, ancho CV (Powell)":
        f = fit_llrg(Xs, ys, semilla, False)
        r = minimize(f, np.zeros(n), method="Powell", bounds=[lw] * n,
                     options=dict(maxfev=presupuesto, xtol=1e-3, ftol=1e-6))
        theta = np.r_[np.ones(n), np.exp(r.x)]; fb = r.fun
    elif metodo == "LLR-gauss, pesos (CMA-ES)":
        f = fit_llrg(Xs, ys, semilla, False)
        u, fb = L.cma_buscar(f, np.zeros(n), np.full(n, lw[0]), np.full(n, lw[1]), semilla, presupuesto)
        theta = np.r_[np.ones(n), np.exp(u)]
    elif metodo == "LLR-gauss, YJ+pesos (CMA-ES)":
        f = fit_llrg(Xs, ys, semilla, True)
        x0 = np.r_[np.ones(n), np.zeros(n)]
        lo = np.r_[np.full(n, -2.0), np.full(n, lw[0])]; hi = np.r_[np.full(n, 2.0), np.full(n, lw[1])]
        u, fb = L.cma_buscar(f, x0, lo, hi, semilla, presupuesto)
        theta = np.r_[u[:n], np.exp(u[n:])]
    elif metodo == "LLR-gauss, pesos (gradiente)":
        e = L.estandar(Xs); a = np.sqrt(2 * L.metrica_kernel(e(Xs), ys, semilla=semilla))   # exp(-d²) == exp(-0.5 (√2 d)²)
        theta = np.r_[np.ones(n), a]; fb = np.nan     # metrica_kernel ya trabaja sobre datos estandarizados
    seg = time.time() - t0
    v = L.ajustar(Xt, theta)
    mae = float(np.mean(np.abs(llr_gauss(v(Xt), yt, v(Xv)) - yv)))
    return dict(dataset=nombre, semilla=semilla, modo=metodo, mae=mae, fitness=fb, k=KG, q=0.0, segundos=seg)
