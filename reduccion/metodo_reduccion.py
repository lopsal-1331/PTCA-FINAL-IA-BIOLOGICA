"""
metodo_reduccion.py — Variante de P en la que la REDUCCIÓN de dimensionalidad la decide el algoritmo evolutivo.

Genoma θ = (λ_1..λ_n, w_1..w_n, τ [, ρ]):
    λ_j ∈ [-2, 2]   exponente Yeo-Johnson (forma)                       — como en final/metodo.py
    w_j ∈ [0, 3]    peso (relevancia)                                   — como en final/metodo.py
    τ   ∈ [0, 3]    UMBRAL DE ELIMINACIÓN aprendido: se eliminan los atributos con w_j < τ
    ρ   ∈ [0, 1]    (opcional) fracción del hueco libre que se rellena con componentes principales (síntesis)

Decodificación (garantiza dim X' < n para CUALQUIER θ, sin reparar ni descartar individuos):
    S = { j : w_j ≥ τ }; si |S| = n se elimina además el de menor peso; si |S| = 0 se conserva el de mayor peso.
    m = round(ρ · (n − 1 − |S|))     componentes principales de z_S que se añaden
    X' = [ z_S | p_1 … p_m ]         con |S| + m ≤ n − 1

Fitness (lo que minimiza CMA-ES):  f(θ) = VC-MAE(KNN, X') / MAE(mediana)  +  α · dim(X') / n
    α = 0 → la única presión para reducir es la restricción s < n; α > 0 premia además usar menos atributos.
"""
import os
import sys
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final"))
import metodo                                                       # noqa: E402

LIM_TAU = (0.0, 3.0)


class TransformacionReducida:
    def __init__(self, theta, n):
        self.theta, self.n = np.asarray(theta, float), n

    def fit(self, Xt):
        n, th = self.n, self.theta
        self.lam, self.w, self.tau = th[:n], th[n:2 * n], float(th[2 * n])
        self.rho = float(th[2 * n + 1]) if len(th) > 2 * n + 1 else 0.0
        S = np.where(self.w >= self.tau)[0]
        if len(S) == n:                                             # restricción s < n: fuera el de menor peso
            S = np.delete(S, np.argmin(self.w))
        if len(S) == 0:
            S = np.array([np.argmax(self.w)])
        self.S = S
        T = metodo.yeo_johnson(Xt[:, S], self.lam[S])
        self.mu, self.sd = T.mean(0), T.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = self._z(Xt)
        self.m = int(round(self.rho * (n - 1 - len(S))))
        self.zm = Z.mean(0)
        self.V = np.linalg.svd(Z - self.zm, full_matrices=False)[2].T[:, :self.m] if self.m else None
        return self

    def _z(self, X):
        return self.w[self.S] * (metodo.yeo_johnson(X[:, self.S], self.lam[self.S]) - self.mu) / self.sd

    def transform(self, X):
        Z = self._z(X)
        return np.hstack([Z, (Z - self.zm) @ self.V]) if self.m else Z

    @property
    def dim(self):
        return len(self.S) + self.m


def hacer_fitness(Xs, ys, semilla, alfa, k=5, pliegues=5):
    escala = np.mean(np.abs(ys - np.median(ys)))
    plie = list(KFold(pliegues, shuffle=True, random_state=semilla).split(Xs))
    n = Xs.shape[1]

    def f(theta):
        try:
            T = TransformacionReducida(theta, n).fit(Xs)
            Z = T.transform(Xs)
            e = np.mean([np.mean(np.abs(metodo.knn_rapido(Z[a], ys[a], Z[b], k) - ys[b])) for a, b in plie]) / escala
            e += alfa * T.dim / n
            return e if np.isfinite(e) else 1e6
        except Exception:
            return 1e6
    return f


def aprender(Xt, yt, semilla=0, alfa=0.0, sintesis=False, presupuesto=1000, n_fit=2000, tau0=0.5, rho0=0.5):
    """P con reducción aprendida. Solo ve el entrenamiento. Arranca en la estandarización (λ=1, w=1)."""
    n = Xt.shape[1]
    idx = np.random.default_rng(semilla).choice(len(yt), min(n_fit, len(yt)), replace=False)
    f = hacer_fitness(Xt[idx], yt[idx], semilla, alfa)
    lo, hi, theta0 = metodo.limites(n)
    lo, hi, theta0 = np.r_[lo, LIM_TAU[0]], np.r_[hi, LIM_TAU[1]], np.r_[theta0, tau0]
    if sintesis:
        lo, hi, theta0 = np.r_[lo, 0.0], np.r_[hi, 1.0], np.r_[theta0, rho0]
    theta, fit, curva = metodo.buscar_cma(f, lo, hi, theta0, presupuesto, semilla)
    P = TransformacionReducida(theta, n).fit(Xt)
    P.fitness, P.fitness0, P.curva = fit, curva[0], curva
    return P
