"""
metodo_v2.py — P v2: aumentar y luego reducir, con fitness de varios modelos y parada por sobreajuste.

1. AUMENTAR. Se construye un conjunto de 2n candidatos:
       c_1..c_n        = ψ(x_j; λ_j) estandarizado                         (los atributos originales, con forma)
       c_{n+1}..c_{2n} = componentes principales de [c_1..c_n], estandarizadas (síntesis: direcciones oblicuas)
   Las componentes se calculan sobre el espacio con forma ANTES de ponderar: no hay dependencia circular y es barato.
2. PONDERAR. Cada candidato tiene su propio peso v_k ∈ [0, 3] ("pesos dobles": n para originales, n para
   componentes).
3. REDUCIR. Se conservan los candidatos con v_k ≥ τ; si son n o más, solo los n−1 de mayor peso; si no queda ninguno,
   el de mayor peso. X' = [v_k · c_k]_{k∈S}, con dim X' ≤ n − 1 para CUALQUIER θ.

Genoma θ = (λ_1..λ_n, v_1..v_2n, τ): 3n + 1 genes. Inicio θ0: λ = 1, v = 1 para originales, v = 0 para componentes,
τ = 0,5 (estandarizar sin componentes, menos el atributo de menor peso).

Fitness = media, sobre los modelos de `MODELOS_FITNESS`, del MAE de validación cruzada (5 pliegues) normalizado por
el MAE de predecir la mediana. Por defecto KNN (k=5), regresión lineal y un árbol de decisión (sustituto barato de
Random Forest, que solo se mide al final).

Parada por sobreajuste: se aparta un 20 % de la submuestra de entrenamiento que el fitness no ve. En cada generación
se evalúa ahí el mejor individuo de la generación; se devuelve el θ con mejor error en esa parte apartada y se para si
no mejora en `paciencia` generaciones (o al agotar el presupuesto).
"""
import os
import sys
import numpy as np
import cma
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(RAIZ, "final"), os.path.join(RAIZ, "reduccion")]
import metodo                                                       # noqa: E402

LIM_LAMBDA, LIM_PESO, LIM_TAU = (-2.0, 2.0), (0.0, 3.0), (0.0, 3.0)


# --------------------------------------------------------------------------------------------------------- modelos
def _knn(A, ya, B):
    return metodo.knn_rapido(A, ya, B, 5)


def _rl(A, ya, B):
    A1 = np.c_[np.ones(len(A)), A]
    coef = np.linalg.lstsq(A1, ya, rcond=None)[0]
    return np.c_[np.ones(len(B)), B] @ coef


def _arbol(A, ya, B):
    return DecisionTreeRegressor(random_state=0).fit(A, ya).predict(B)


def _rf(A, ya, B, arboles=5):
    """Random Forest reducido (5 árboles) para que el fitness sea asequible (~130 ms por evaluación, como el de
    KNN+RL+árbol); con 20 árboles costaría ~500 ms. El RF de evaluación final tiene 100 árboles."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(arboles, random_state=0, n_jobs=1).fit(A, ya).predict(B)


PREDICTORES = {"KNN": _knn, "RL": _rl, "Árbol": _arbol, "RF": _rf}

# P configurable por familia: el fitness usa el modelo de la familia para la que se aprende X'.
# "todos" = una sola X' para los cuatro (fitness KNN + RL + árbol), la P v2 original.
FAMILIAS = {"KNN": ("KNN",), "RL": ("RL",), "Árbol": ("Árbol",), "RF": ("RF",), "todos": ("KNN", "RL", "Árbol")}


# --------------------------------------------------------------------------------------------------- transformación
class TransformacionV2:
    def __init__(self, theta, n):
        self.theta, self.n = np.asarray(theta, float), n

    def fit(self, Xt):
        n, th = self.n, self.theta
        self.lam, self.v, self.tau = th[:n], th[n:3 * n], float(th[3 * n])
        T = metodo.yeo_johnson(Xt, self.lam)
        self.mu, self.sd = T.mean(0), T.std(0)
        self.sd[self.sd == 0] = 1.0
        C = (T - self.mu) / self.sd
        self.V = np.linalg.svd(C, full_matrices=False)[2].T          # componentes del espacio con forma
        P = C @ self.V
        self.psd = P.std(0)
        self.psd[self.psd < 1e-12] = 1.0
        S = np.where(self.v >= self.tau)[0]
        if len(S) >= n:                                              # restricción: como mucho n − 1
            S = np.sort(S[np.argsort(-self.v[S])[:n - 1]])
        if len(S) == 0:
            S = np.array([np.argmax(self.v)])
        self.S = S
        return self

    def candidatos(self, X):
        C = (metodo.yeo_johnson(X, self.lam) - self.mu) / self.sd
        return np.hstack([C, (C @ self.V) / self.psd])

    def transform(self, X):
        return self.v[self.S] * self.candidatos(X)[:, self.S]

    @property
    def dim(self):
        return len(self.S)

    @property
    def n_orig(self):
        return int((self.S < self.n).sum())

    @property
    def n_comp(self):
        return int((self.S >= self.n).sum())


# ---------------------------------------------------------------------------------------------------------- fitness
def error_normalizado(Za, ya, Zb, yb, modelos):
    escala = np.mean(np.abs(yb - np.median(ya)))
    return np.mean([np.mean(np.abs(PREDICTORES[m](Za, ya, Zb) - yb)) / escala for m in modelos])


def errores_cv(Z, ys, plie, modelos):
    """Vector con el MAE de validación cruzada de cada modelo, normalizado por el de predecir la mediana."""
    r = np.zeros(len(modelos))
    for a, b in plie:
        escala = np.mean(np.abs(ys[b] - np.median(ys[a])))
        r += [np.mean(np.abs(PREDICTORES[m](Z[a], ys[a], Z[b]) - ys[b])) / escala for m in modelos]
    return r / len(plie)


AGREGACIONES = ("media", "rel-media", "rel-max", "rel-mix")


def hacer_fitness(Xs, ys, semilla, modelos, pliegues=5, agregacion="media"):
    """agregacion:
        media      media de los errores normalizados por la mediana (P v2)
        rel-media  media de r_m = error_m(θ) / error_m(X estandarizado): cada modelo frente a sí mismo
        rel-max    max_m r_m (minimax): solo mejora si mejoran TODOS los modelos
        rel-mix    (max_m r_m + media_m r_m) / 2"""
    plie = list(KFold(pliegues, shuffle=True, random_state=semilla).split(Xs))
    n = Xs.shape[1]
    base = None
    if agregacion != "media":
        Z0 = (Xs - Xs.mean(0)) / np.where(Xs.std(0) > 0, Xs.std(0), 1)      # referencia: X estandarizado completo
        base = errores_cv(Z0, ys, plie, modelos)

    def f(theta):
        try:
            Z = TransformacionV2(theta, n).fit(Xs).transform(Xs)
            r = errores_cv(Z, ys, plie, modelos)
            if base is not None:
                r = r / base
            e = {"media": r.mean(), "rel-media": r.mean(), "rel-max": r.max(),
                 "rel-mix": 0.5 * (r.max() + r.mean())}[agregacion]
            return e if np.isfinite(e) else 1e6
        except Exception:
            return 1e6
    return f


def hacer_validacion(Xa, ya, Xb, yb, modelos):
    """Error en la parte apartada: la transformación y los modelos se ajustan con Xa y se miden en Xb."""
    n = Xa.shape[1]

    def g(theta):
        try:
            T = TransformacionV2(theta, n).fit(Xa)
            return error_normalizado(T.transform(Xa), ya, T.transform(Xb), yb, modelos)
        except Exception:
            return 1e6
    return g


# ---------------------------------------------------------------------------------------------------------- búsqueda
def limites(n):
    lo = np.r_[np.full(n, LIM_LAMBDA[0]), np.full(2 * n, LIM_PESO[0]), LIM_TAU[0]]
    hi = np.r_[np.full(n, LIM_LAMBDA[1]), np.full(2 * n, LIM_PESO[1]), LIM_TAU[1]]
    theta0 = np.r_[np.ones(n), np.ones(n), np.zeros(n), 0.5]
    return lo, hi, theta0


def aprender(Xt, yt, semilla=0, modelos=("KNN", "RL", "Árbol"), parada=False, presupuesto=1000, n_fit=2000,
             apartado=0.2, paciencia=15, sigma0=0.2, agregacion="media", familia=None):
    """P v2. Solo ve el entrenamiento. Devuelve la transformación ajustada con todo Xt.
    familia: "KNN", "RL", "Árbol", "RF" o "todos" -> el fitness usa el modelo de esa familia (FAMILIAS); si se da,
    sustituye a `modelos`."""
    if familia is not None:
        modelos = FAMILIAS[familia]
    n = Xt.shape[1]
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(yt), min(n_fit, len(yt)), replace=False)
    Xs, ys = Xt[idx], yt[idx]
    if parada:
        corte = int(round(len(idx) * (1 - apartado)))
        Xa, ya, Xb, yb = Xs[:corte], ys[:corte], Xs[corte:], ys[corte:]
        f, g = hacer_fitness(Xa, ya, semilla, modelos, agregacion=agregacion), hacer_validacion(Xa, ya, Xb, yb, modelos)
    else:
        f, g = hacer_fitness(Xs, ys, semilla, modelos, agregacion=agregacion), None
    lo, hi, theta0 = limites(n)
    dec = lambda u: lo + np.clip(u, 0, 1) * (hi - lo)
    es = cma.CMAEvolutionStrategy((theta0 - lo) / (hi - lo), sigma0,
                                  dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
    evals, mejor_f, mejor_g, mejor_u, sin_mejora, curva = 0, np.inf, np.inf, None, 0, []
    while not es.stop():
        U = es.ask()
        F = [f(dec(u)) for u in U]
        es.tell(U, F)
        evals += len(U)
        i = int(np.argmin(F))
        if g is None:
            if F[i] < mejor_f:
                mejor_f, mejor_u = F[i], U[i]
            curva.append(mejor_f)
            continue
        gi = g(dec(U[i]))
        curva.append((F[i], gi))
        if gi < mejor_g - 1e-4:
            mejor_g, mejor_f, mejor_u, sin_mejora = gi, F[i], U[i], 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    theta = dec(mejor_u)
    P = TransformacionV2(theta, n).fit(Xt)
    P.fitness, P.validacion, P.evaluaciones, P.curva = mejor_f, mejor_g, evals, curva
    return P
