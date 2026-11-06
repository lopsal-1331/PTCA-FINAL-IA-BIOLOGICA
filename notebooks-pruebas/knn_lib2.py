"""Exploración: métrica aprendida (CMA-ES) para distintas reglas de predicción tipo KNN."""
import numpy as np, time, warnings
import knn_lib as L
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings("ignore")

MODOS = ["mean_k5", "mean", "wdist", "llr", "res_mean", "res_llr"]

def lr_ajustar(Za, ya):
    A = np.c_[np.ones(len(Za)), Za]
    return np.linalg.lstsq(A, ya, rcond=None)[0]

def predecir(Za, ya, Zb, k, modo, q=0.0, trozo=400):
    """Predice Zb con referencia (Za, ya). q = exponente de distancia (wdist) o log-ridge (llr)."""
    base_b = 0.0
    if modo.startswith("res"):
        beta = lr_ajustar(Za, ya)
        ya = ya - np.c_[np.ones(len(Za)), Za] @ beta
        base_b = np.c_[np.ones(len(Zb)), Zb] @ beta
    out = np.empty(len(Zb)); k = min(k, len(Za)); na = (Za ** 2).sum(1)
    for i in range(0, len(Zb), trozo):
        zb = Zb[i:i + trozo]
        d2 = (zb ** 2).sum(1)[:, None] + na[None, :] - 2 * zb @ Za.T
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dk = np.maximum(np.take_along_axis(d2, idx, 1), 0)
        yn = ya[idx]
        if modo in ("mean_k5", "mean", "res_mean"):
            p = yn.mean(1)
        elif modo == "wdist":
            w = 1.0 / (np.sqrt(dk) + 1e-6) ** q
            p = (w * yn).sum(1) / w.sum(1)
        else:  # llr / res_llr : regresión lineal local ponderada con ridge
            u = np.sqrt(dk / (dk.max(1, keepdims=True) + 1e-12))
            w = (1 - u ** 3) ** 3 + 1e-3
            Dz = Za[idx] - zb[:, None, :]
            A = np.concatenate([np.ones(Dz.shape[:2] + (1,)), Dz], 2)
            Aw = A * w[..., None]
            M = np.einsum("qki,qkj->qij", Aw, A)
            reg = np.exp(q) * w.sum(1)[:, None, None] * np.eye(A.shape[2])[None]
            reg[:, 0, 0] = 0
            b = np.einsum("qki,qk->qi", Aw, yn)
            try:
                p = np.linalg.solve(M + reg + 1e-9 * np.eye(A.shape[2]), b[..., None])[:, 0, 0]
            except np.linalg.LinAlgError:
                p = yn.mean(1)
        out[i:i + trozo] = p
    return out + base_b

def estructura(modo, n):
    """Límites de theta = (lambdas, pesos, [log k], [q])."""
    lo = list(np.full(n, -2.0)) + list(np.zeros(n)); hi = list(np.full(n, 2.0)) + list(np.full(n, 3.0))
    x0 = list(np.ones(2 * n))
    kmin = n + 3 if "llr" in modo else 2
    if modo != "mean_k5":
        lo += [np.log(kmin)]; hi += [np.log(80)]; x0 += [np.log(max(kmin, 10 if "llr" in modo else 5))]
    if modo == "wdist":   lo += [0.0]; hi += [4.0]; x0 += [1.0]
    if "llr" in modo:     lo += [np.log(1e-3)]; hi += [np.log(10.0)]; x0 += [np.log(0.1)]
    return np.array(x0), np.array(lo), np.array(hi)

def separar(theta, modo, n):
    theta = np.asarray(theta, float); t = theta[:2 * n]; r = theta[2 * n:]
    k, q = 5, 0.0
    if modo != "mean_k5": k = int(round(np.exp(r[0]))); r = r[1:]
    if modo == "wdist": q = r[0]
    if "llr" in modo: q = r[0]
    return t, k, q

def fitness(Xs, ys, semilla, modo):
    n = Xs.shape[1]; escala = np.mean(np.abs(ys - np.median(ys)))
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Xs))
    def f(theta):
        try:
            t, k, q = separar(theta, modo, n)
            Z = L.ajustar(Xs, t)(Xs); tot = 0.0
            for a, b in plie:
                tot += np.mean(np.abs(predecir(Z[a], ys[a], Z[b], k, modo, q) - ys[b]))
            r = tot / 5 / escala
            return r if np.isfinite(r) else 1e6
        except Exception:
            return 1e6
    return f

def trabajo(nombre, semilla, modo, raiz="../", presupuesto=2020):
    X, y = L.cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    f = fitness(Xt[sub], yt[sub], semilla, modo)
    x0, lo, hi = estructura(modo, n)
    t0 = time.time(); th, fb = L.cma_buscar(f, x0, lo, hi, semilla, presupuesto); seg = time.time() - t0
    t, k, q = separar(th, modo, n)
    v = L.ajustar(Xt, t)
    mae = float(np.mean(np.abs(predecir(v(Xt), yt, v(Xv), k, modo, q) - yv)))
    return dict(dataset=nombre, semilla=semilla, modo=modo, mae=mae, fitness=fb, k=k, q=q, segundos=seg)

# ---------------------------------------------------------------- regularización
ALFA = 0.1
def trabajo2(nombre, semilla, modo, var, raiz="../", presupuesto=2020):
    """var: 'reg' (penaliza alejarse de la identidad), 'comp' (un exponente compartido), 'reg+comp'."""
    X, y = L.cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    f0 = fitness(Xt[sub], yt[sub], semilla, modo)
    x0, lo, hi = estructura(modo, n)
    comp, reg = "comp" in var, "reg" in var
    if comp:
        expand = lambda t: np.r_[np.full(n, t[0]), t[1:]]
        x0r, lor, hir = np.r_[x0[0], x0[n:]], np.r_[lo[0], lo[n:]], np.r_[hi[0], hi[n:]]
    else:
        expand = lambda t: np.asarray(t, float)
        x0r, lor, hir = x0, lo, hi
    def f(t):
        th = expand(t); v = f0(th)
        if reg:
            dist = np.mean(np.abs((th[:2 * n] - x0[:2 * n]) / (hi[:2 * n] - lo[:2 * n])))
            v += ALFA * dist
        return v
    t0 = time.time(); th, fb = L.cma_buscar(f, x0r, lor, hir, semilla, presupuesto); seg = time.time() - t0
    th = expand(th); t, k, q = separar(th, modo, n)
    v = L.ajustar(Xt, t)
    mae = float(np.mean(np.abs(predecir(v(Xt), yt, v(Xv), k, modo, q) - yv)))
    return dict(dataset=nombre, semilla=semilla, modo=f"{modo}+{var}", mae=mae, fitness=fb, k=k, q=q, segundos=seg)
