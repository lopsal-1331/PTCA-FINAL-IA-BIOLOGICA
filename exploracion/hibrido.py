"""Híbrido: CMA-ES + fitness estocástico (pliegues nuevos por generación) + inyección tipo DE + selección final robusta."""
import numpy as np, time, cma
from sklearn.model_selection import KFold, train_test_split
import lib

def fitness_rep(Xs, ys, semilla):
    escala = np.mean(np.abs(ys - np.median(ys))); cache = {}
    def pliegues(rep):
        if rep not in cache:
            cache[rep] = list(KFold(5, shuffle=True, random_state=semilla * 1000 + rep).split(Xs))
        return cache[rep]
    cuenta = [0]
    def f(theta, rep=0):
        cuenta[0] += 1
        try:
            Z = lib.ajustar(Xs, np.asarray(theta, float))(Xs); tot = 0.0
            for a, b in pliegues(rep):
                tot += np.mean(np.abs(lib.lineal(Z[a], ys[a], Z[b]) - ys[b]))
                tot += np.mean(np.abs(lib.knn5(Z[a], ys[a], Z[b]) - ys[b]))
            r = tot / 10 / escala
            return r if np.isfinite(r) else 1e6
        except Exception:
            return 1e6
    return f, cuenta

def buscar(f, lo, hi, semilla, variante, presupuesto=1000, F=0.5, CR=0.5, margen=0.005):
    n2 = len(lo); th0 = np.ones(n2)
    estoc = "folds" in variante; de = "DE" in variante; sel = "sel" in variante
    reserva = 100 if sel else 0
    to_th = lambda u: lo + np.clip(u, 0, 1) * (hi - lo)
    rng = np.random.default_rng(semilla)
    es = cma.CMAEvolutionStrategy((th0 - lo) / (hi - lo), 0.2,
            dict(seed=semilla + 1, bounds=[0, 1], verbose=-9, tolfun=1e-12, tolx=1e-12))
    usados = 0; g = 0; pool = []
    while usados < presupuesto - reserva:
        sols = es.ask(); rep = g if estoc else 0
        fv = [f(to_th(u), rep) for u in sols]; usados += len(sols)
        es.tell(sols, fv)
        pool.append((to_th(es.mean), to_th(sols[int(np.argmin(fv))]), min(fv)))
        if de and len(sols) >= 4:
            orden = np.argsort(fv)[:max(3, len(sols) // 2)]            # padres (mitad superior)
            r1, r2, r3 = rng.choice(orden, 3, replace=False)
            mejor = sols[orden[0]]; v = np.array(sols[r1]) + F * (np.array(sols[r2]) - np.array(sols[r3]))
            m = rng.random(n2) < CR; m[rng.integers(n2)] = True
            es.inject([np.clip(np.where(m, v, mejor), 0, 1)])
        g += 1
    if not sel:
        return to_th(es.result.xbest if not estoc else es.mean)
    # selección robusta: candidatos = medias y mejores de las últimas generaciones (+ θ0), pliegues nuevos
    cand = [th0]
    for mu_, b_, _ in pool[-8:]: cand += [mu_, b_]
    cand += [min(pool, key=lambda p: p[2])[1]]
    R = max(1, reserva // len(cand)); reps = [10_000 + i for i in range(R)]
    pun = [np.mean([f(c, r) for r in reps]) for c in cand]
    i = int(np.argmin(pun))
    return cand[i] if pun[i] < pun[0] - margen else th0

VARIANTES = ["CMA-ES (base)", "CMA+folds", "CMA+folds+sel", "CMA+DE", "Híbrido (folds+DE+sel)"]
def correr(nombre, semilla, variante, raiz="../", presupuesto=1000):
    X, y = lib.cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    f, cuenta = fitness_rep(Xt[sub], yt[sub], semilla)
    lo = np.r_[np.full(n, -2.0), np.zeros(n)]; hi = np.r_[np.full(n, 2.0), np.full(n, 3.0)]
    t0 = time.time(); th = buscar(f, lo, hi, semilla, variante, presupuesto); seg = time.time() - t0
    v = lib.ajustar(Xt, th)
    d = dict(dataset=nombre, semilla=semilla, algoritmo=variante, fitness=f(th, 0), evaluaciones=cuenta[0] - 1, segundos=seg,
             cambio_theta=float(np.mean(np.abs(th - 1)) ))
    d.update(lib.evaluar(v(Xt), v(Xv), yt, yv))
    d.update({"crudo " + k: val for k, val in lib.evaluar(Xt, Xv, yt, yv).items()})
    return d
