"""
ptca.py — El programa P de la práctica y todo lo necesario para evaluarlo.

ENUNCIADO. A partir SOLO del entrenamiento (Xt, yt), P aprende una transformación X -> X' que
  (a) hace síntesis de atributos y reducción de la dimensionalidad (dim X' < n), con un criterio de eliminación que
      depende del algoritmo evolutivo, y
  (b) consigue que un modelo arbitrario M entrenado con X' tenga menor MAE en validación que con X (e' < e).

MÉTODO: aumentar, ponderar y reducir.
  1. Forma.      x~_j = (ψ(x_j; λ_j) − μ_j) / σ_j        Yeo-Johnson con λ_j ∈ [−2, 2], estandarizado.
  2. Aumentar.   c = [x~_1 .. x~_n | p_1 .. p_n]          p = componentes principales de x~ (estandarizadas):
                                                          2n candidatos, los originales y n direcciones oblicuas.
  3. Ponderar.   v_k · c_k,  v_k ∈ [0, 3]                  un peso por candidato ("pesos dobles").
  4. Reducir.    se conservan los c_k con v_k ≥ τ (τ ∈ [0, 3]); si son n o más, los n−1 de mayor peso; si no queda
                 ninguno, el de mayor peso. => dim X' ≤ n − 1 para CUALQUIER individuo.
  Genoma θ = (λ_1..λ_n, v_1..v_2n, τ), 3n + 1 genes reales. CMA-ES decide qué candidatos sobreviven y cuántos.

FITNESS: media, sobre KNN (k=5), regresión lineal y un árbol de decisión, del MAE de validación cruzada (5 pliegues
fijos) dividido por el MAE de predecir la mediana, en una submuestra de hasta 2000 filas del entrenamiento.

BÚSQUEDA: CMA-ES (librería `cma`) en el cubo [0, 1]^D reescalado a los límites, σ0 = 0,2, población por defecto,
1000 evaluaciones, inicio en θ0 = estandarizar sin componentes (λ = 1, v_orig = 1, v_comp = 0, τ = 0,5).

`aprender()` admite variantes (fitness, síntesis, límite, forma, optimizador, parada) usadas SOLO en el cribado de
diseño y en las ablaciones que justifican cada decisión; los valores por defecto son el método final.
"""
import os
import warnings
import numpy as np
import pandas as pd
import cma
from scipy.stats import yeojohnson, wilcoxon, friedmanchisquare, studentized_range
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
CARPETA = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================================ datos
PRACTICA = {"diabetes": ("datos/Diabetes.csv", "target"), "california": ("datos/California.csv", "MedHouseVal")}
DATASETS = ["diabetes", "california", "concrete_compressive_strength", "stock", "space_ga", "sulfur",
            "kin8nm", "bank8FM", "puma8NH", "house_8L", "cpu_small", "wine_quality"]


def cargar(nombre, max_filas=10_000):
    """(X, y) como arrays float: atributos numéricos no constantes; como mucho 10 000 filas (submuestra fija).
    Los diez datasets de OpenML se descargan la primera vez y se guardan en ~/scikit_learn_data."""
    if nombre in PRACTICA:
        ruta, objetivo = PRACTICA[nombre]
        df = pd.read_csv(os.path.join(CARPETA, ruta))
        return df.drop(columns=objetivo).values.astype(float), df[objetivo].values.astype(float)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name=nombre, as_frame=True, parser="auto")
    X = d.data.select_dtypes("number").dropna(axis=1).astype(float)
    X = X.loc[:, X.nunique() > 1].values
    y = d.target.astype(float).values
    if len(y) > max_filas:
        i = np.random.default_rng(0).choice(len(y), max_filas, replace=False)
        X, y = X[i], y[i]
    return X, y


def particion(nombre, semilla):
    """Partición 70/30. Las semillas 40–44 son las de confirmación; las 0–1, las de exploración (cribado)."""
    X, y = cargar(nombre)
    return train_test_split(X, y, test_size=0.3, random_state=semilla)


def modelos(semilla):
    """Los cuatro modelos triviales y fijos con los que se mide e y e' (parámetros por defecto)."""
    return {"Regresión lineal": LinearRegression(), "KNN": KNeighborsRegressor(5),
            "Árbol": DecisionTreeRegressor(random_state=semilla),
            "Random Forest": RandomForestRegressor(100, random_state=semilla, n_jobs=1)}


# ======================================================================================================= transformación
LIM_LAMBDA, LIM_PESO, LIM_TAU = (-2.0, 2.0), (0.0, 3.0), (0.0, 3.0)


def yeo_johnson(X, lam):
    return np.column_stack([yeojohnson(X[:, j], lmbda=lam[j]) for j in range(X.shape[1])])


class Transformacion:
    """X' = P(X). Se ajusta con el entrenamiento (μ, σ y componentes) y se aplica igual a entrenamiento y validación.
    componentes=False: sin síntesis (solo los n originales). limite=False: sin el tope de n−1 (ablación)."""

    def __init__(self, theta, n, componentes=True, limite=True):
        self.theta, self.n, self.componentes, self.limite = np.asarray(theta, float), n, componentes, limite

    def fit(self, Xt):
        n, th, k = self.n, self.theta, (2 if self.componentes else 1) * self.n
        self.lam, self.v, self.tau = th[:n], th[n:n + k], float(th[n + k])
        T = yeo_johnson(Xt, self.lam)
        self.mu, self.sd = T.mean(0), T.std(0)
        self.sd[self.sd == 0] = 1.0
        C = (T - self.mu) / self.sd
        self.V = np.linalg.svd(C, full_matrices=False)[2].T          # componentes del espacio con forma
        self.psd = (C @ self.V).std(0)
        self.psd[self.psd < 1e-12] = 1.0
        S = np.where(self.v >= self.tau)[0]
        if self.limite and len(S) >= n:                              # reducción: como mucho n − 1
            S = np.sort(S[np.argsort(-self.v[S])[:n - 1]])
        if len(S) == 0:
            S = np.array([np.argmax(self.v)])
        self.S = S
        return self

    def candidatos(self, X):
        C = (yeo_johnson(X, self.lam) - self.mu) / self.sd
        return np.hstack([C, (C @ self.V) / self.psd]) if self.componentes else C

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


# ============================================================================================================== fitness
def _knn(A, ya, B, k=5):
    """KNN por fuerza bruta en NumPy (igual que KNeighborsRegressor salvo empates); se llama miles de veces."""
    d = (B ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2 * B @ A.T
    return ya[np.argpartition(d, k - 1, axis=1)[:, :k]].mean(1)


def _rl(A, ya, B):
    coef = np.linalg.lstsq(np.c_[np.ones(len(A)), A], ya, rcond=None)[0]
    return np.c_[np.ones(len(B)), B] @ coef


def _arbol(A, ya, B):
    return DecisionTreeRegressor(random_state=0).fit(A, ya).predict(B)


def _rf(A, ya, B):
    """RF reducido (5 árboles) para el cribado 'por familia': con 20 árboles el fitness costaría ~12 veces más."""
    return RandomForestRegressor(5, random_state=0, n_jobs=1).fit(A, ya).predict(B)


PREDICTORES = {"KNN": _knn, "RL": _rl, "Árbol": _arbol, "RF": _rf}
FITNESS_FINAL = ("KNN", "RL", "Árbol")


def errores_cv(Z, ys, plie, fitness):
    """MAE de validación cruzada de cada modelo del fitness, normalizado por el de predecir la mediana."""
    r = np.zeros(len(fitness))
    for a, b in plie:
        escala = np.mean(np.abs(ys[b] - np.median(ys[a])))
        r += [np.mean(np.abs(PREDICTORES[m](Z[a], ys[a], Z[b]) - ys[b])) / escala for m in fitness]
    return r / len(plie)


def hacer_fitness(Xs, ys, semilla, fitness=FITNESS_FINAL, agregacion="media", **opciones):
    """f(θ) que minimiza la búsqueda. agregacion='media' (método) o 'minimax' (cribado: el peor modelo, cada uno
    relativo a su error con X estandarizado)."""
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Xs))
    n = Xs.shape[1]
    base = None
    if agregacion == "minimax":
        base = errores_cv((Xs - Xs.mean(0)) / np.where(Xs.std(0) > 0, Xs.std(0), 1), ys, plie, fitness)

    def f(theta):
        try:
            Z = Transformacion(theta, n, **opciones).fit(Xs).transform(Xs)
            r = errores_cv(Z, ys, plie, fitness)
            e = r.mean() if base is None else (r / base).max()
            return e if np.isfinite(e) else 1e6
        except Exception:                                           # λ extremos pueden desbordar: inválido
            return 1e6
    return f


# ============================================================================================================= búsqueda
def genoma(n, componentes=True, forma=True):
    """Límites y punto de partida θ0 (estandarizar sin componentes). Con forma=False, λ = 1 fijo (fuera del genoma)."""
    k = (2 if componentes else 1) * n
    lo = np.r_[np.full(n, LIM_LAMBDA[0]), np.full(k, LIM_PESO[0]), LIM_TAU[0]]
    hi = np.r_[np.full(n, LIM_LAMBDA[1]), np.full(k, LIM_PESO[1]), LIM_TAU[1]]
    th0 = np.r_[np.ones(n), np.ones(n), np.zeros(k - n), 0.5]
    libres = np.arange(len(lo)) if forma else np.arange(n, len(lo))
    return lo, hi, th0, libres


def cma_es(f, D, u0, semilla, presupuesto, g=None, paciencia=None):
    """CMA-ES en [0,1]^D. Devuelve (u*, f(u*), evaluaciones). Con g (error en datos apartados) devuelve el u con mejor
    g y, si hay paciencia, para cuando g no mejora en `paciencia` generaciones (variante de parada, descartada)."""
    es = cma.CMAEvolutionStrategy(u0, 0.2, dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
    evals, mejor, mejor_g, mejor_u, sin_mejora = 0, np.inf, np.inf, None, 0
    while not es.stop():
        U = es.ask()
        F = [f(u) for u in U]
        es.tell(U, F)
        evals += len(U)
        i = int(np.argmin(F))
        if g is None:
            if F[i] < mejor:
                mejor, mejor_u = F[i], U[i]
            continue
        gi = g(U[i])
        if gi < mejor_g - 1e-4:
            mejor_g, mejor, mejor_u, sin_mejora = gi, F[i], U[i], 0
        else:
            sin_mejora += 1
            if paciencia and sin_mejora >= paciencia:
                break
    return mejor_u, mejor, evals


def aleatoria(f, D, u0, semilla, presupuesto):
    """Control: el mejor de θ0 y presupuesto−1 puntos uniformes."""
    rng = np.random.default_rng(semilla)
    mejor_u, mejor = u0, f(u0)
    for _ in range(presupuesto - 1):
        u = rng.uniform(0, 1, D)
        v = f(u)
        if v < mejor:
            mejor_u, mejor = u, v
    return mejor_u, mejor, presupuesto


def ag(f, D, u0, semilla, presupuesto, N=30):
    """Control: algoritmo genético. Torneo de 3, cruce BLX-α (α = 0,3), mutación gaussiana (p = 1/D, σ = 0,1) y
    elitismo de 1; θ0 en la población inicial."""
    rng = np.random.default_rng(semilla)
    P = np.vstack([u0, rng.uniform(0, 1, (N - 1, D))])
    F = np.array([f(p) for p in P])
    usadas = N
    while usadas < presupuesto:
        hijos = []
        while len(hijos) < N - 1 and usadas + len(hijos) < presupuesto:
            i, j = [min(rng.integers(0, len(P), 3), key=lambda k: F[k]) for _ in range(2)]
            c = P[i] + rng.uniform(-0.3, 1.3, D) * (P[j] - P[i])
            c = np.where(rng.random(D) < 1.0 / D, c + rng.normal(0, 0.1, D), c)
            hijos.append(np.clip(c, 0, 1))
        Fh = np.array([f(h) for h in hijos])
        usadas += len(hijos)
        e = np.argmin(F)
        P, F = np.vstack([P[e], hijos]), np.r_[F[e], Fh]
    return P[np.argmin(F)], F.min(), usadas


def aprender(Xt, yt, semilla=0, fitness=FITNESS_FINAL, agregacion="media", componentes=True, limite=True,
             forma=True, optimizador="cma", parada=None, presupuesto=1000, n_fit=2000):
    """EL PROGRAMA P. Solo ve el entrenamiento; devuelve la transformación ajustada con todo Xt.
    Los valores por defecto son el método final. parada: None | ("parada", paciencia) | ("seleccion", None)."""
    n = Xt.shape[1]
    idx = np.random.default_rng(semilla).choice(len(yt), min(n_fit, len(yt)), replace=False)
    Xs, ys = Xt[idx], yt[idx]
    op = dict(componentes=componentes, limite=limite)
    lo, hi, th0, libres = genoma(n, componentes, forma)

    def theta(u):
        t = th0.copy()
        t[libres] = lo[libres] + np.clip(u, 0, 1) * (hi[libres] - lo[libres])
        return t

    g = None
    if parada:                                                       # apartar un 20 % para vigilar el sobreajuste
        corte = int(round(len(idx) * 0.8))
        Xa, ya, Xb, yb = Xs[:corte], ys[:corte], Xs[corte:], ys[corte:]
        fx = hacer_fitness(Xa, ya, semilla, fitness, agregacion, **op)

        def g(u):
            T = Transformacion(theta(u), n, **op).fit(Xa)
            escala = np.mean(np.abs(yb - np.median(ya)))
            Za, Zb = T.transform(Xa), T.transform(Xb)
            return np.mean([np.mean(np.abs(PREDICTORES[m](Za, ya, Zb) - yb)) / escala for m in fitness])
    else:
        fx = hacer_fitness(Xs, ys, semilla, fitness, agregacion, **op)
    f = lambda u: fx(theta(u))
    u0 = (th0[libres] - lo[libres]) / (hi[libres] - lo[libres])
    if optimizador == "cma":
        u, fit, evals = cma_es(f, len(libres), u0, semilla, presupuesto, g, parada[1] if parada else None)
    else:
        u, fit, evals = {"aleatoria": aleatoria, "ag": ag}[optimizador](f, len(libres), u0, semilla, presupuesto)
    P = Transformacion(theta(u), n, **op).fit(Xt)
    P.fitness, P.evaluaciones = fit, evals
    return P


# =========================================================================================================== estadística
def geo(x):
    """Media geométrica (los cocientes e'/e se promedian en escala logarítmica)."""
    return float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def holm(p):
    """p-valores ajustados por Holm (controla el error de familia)."""
    p = np.asarray(p, float)
    aj, acum = np.empty_like(p), 0.0
    for i, j in enumerate(np.argsort(p)):
        acum = max(acum, (len(p) - i) * p[j])
        aj[j] = min(1.0, acum)
    return aj


def por_dataset(rel):
    """rel: Serie indexada por (dataset, semilla) con cocientes. Unidad estadística = dataset (Demšar, 2006): las
    particiones de un dataset comparten filas y no son independientes."""
    return rel.groupby(level="dataset").apply(geo)


def contraste(rel):
    """Cociente medio (geométrico entre datasets), datasets que mejoran y p de Wilcoxon por dataset."""
    d = por_dataset(rel)
    lg = np.log(d.values)
    p = wilcoxon(lg).pvalue if np.any(np.abs(lg) > 1e-12) else 1.0
    return dict(cociente=geo(d), cambio=100 * (geo(d) - 1), ds_mejora=int((d < 1).sum()), ds=len(d), p=float(p),
                peor_ds=d.idxmax(), peor=float(d.max()))


def friedman_nemenyi(tabla):
    """tabla: datasets × métodos (menor = mejor): rango medio, p de Friedman y diferencia crítica de Nemenyi."""
    rangos = tabla.rank(axis=1).mean()
    N, k = tabla.shape
    p = friedmanchisquare(*[tabla[c].values for c in tabla.columns]).pvalue
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    return rangos.sort_values(), float(p), float(cd)
