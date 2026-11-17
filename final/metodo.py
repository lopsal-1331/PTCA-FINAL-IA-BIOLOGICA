"""
metodo.py — Método final de la práctica: transformación evolutiva de los atributos.

El enunciado pide un programa P que, a partir SOLO del entrenamiento (X_t, y_t), aprenda una transformación
X -> X' tal que un modelo arbitrario M entrenado con X' tenga menos error (MAE) en validación que entrenado con X.

Nuestro P construye X' en tres bloques, uno por cada problema de la representación:

    1. FORMA       ψ(x_j; λ_j)        Yeo-Johnson con un exponente λ_j ∈ [-2, 2] por atributo (corrige colas/asimetría).
    2. RELEVANCIA  z_j = w_j·(ψ_j - μ_j)/σ_j   estandarizar y ponderar con w_j ∈ [0, 3]; w_j ≈ 0 apaga el atributo
                                               (reducción/selección de atributos aprendida).
    3. SÍNTESIS    p = Vᵀ(z - z̄)        componentes principales del espacio YA aprendido, añadidas como atributos nuevos.

    X' = [ z_1 … z_n | p_1 … p_n ]

θ = (λ_1..λ_n, w_1..w_n) se busca con CMA-ES minimizando el MAE de validación cruzada de KNN en el entrenamiento.
El bloque 3 no tiene parámetros propios: sus direcciones dependen de θ.

Por qué el bloque de síntesis es "gratis" para KNN y la regresión lineal pero ayuda a los árboles:
  * V es ortogonal, así que ‖[z, Vᵀz]‖² = 2‖z‖²: las distancias se multiplican por √2 y los vecinos son los mismos.
  * Las componentes son combinaciones lineales de z: la regresión lineal ve el mismo subespacio, misma predicción.
  * Un árbol solo corta en paralelo a los ejes; las componentes le dan cortes oblicuos que no podía construir.
  * YJ y los pesos son monótonos por atributo, así que por sí solos no cambian a un árbol: le llegan a través del bloque 3.
"""
import os
import warnings
import numpy as np
import pandas as pd
import cma
from scipy.stats import yeojohnson
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------------
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # carpeta PTCA-FINAL

# Los dos datasets de la práctica (CSV locales) y diez de regresión de OpenML para comprobar que generaliza
PRACTICA = {"diabetes": ("DATASETS/Diabetes.csv", "target"),
            "california": ("DATASETS/California.csv", "MedHouseVal")}
DATASETS = ["diabetes", "california", "concrete_compressive_strength", "stock", "space_ga", "sulfur",
            "kin8nm", "bank8FM", "puma8NH", "house_8L", "cpu_small", "wine_quality"]


def cargar(nombre, max_filas=10_000):
    """Devuelve (X, y) como arrays float. Solo atributos numéricos y no constantes; como mucho 10 000 filas
    (submuestra fija) para que el coste sea comparable entre datasets. OpenML se cachea en ~/scikit_learn_data."""
    if nombre in PRACTICA:
        ruta, objetivo = PRACTICA[nombre]
        df = pd.read_csv(os.path.join(RAIZ, ruta))
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


# --------------------------------------------------------------------------------------------------
# La transformación
# --------------------------------------------------------------------------------------------------
LIM_LAMBDA = (-2.0, 2.0)     # rango de los exponentes Yeo-Johnson (justificado en experimento_rango_yj.py)
LIM_PESO = (0.0, 3.0)        # rango de los pesos


def yeo_johnson(X, lam):
    """Aplica Yeo-Johnson columna a columna, cada una con su propio exponente."""
    return np.column_stack([yeojohnson(X[:, j], lmbda=lam[j]) for j in range(X.shape[1])])


class Transformacion:
    """Transformación aprendida. Se 'ajusta' con el entrenamiento (medias, desviaciones y componentes se
    calculan SOLO con Xt) y luego se aplica igual a entrenamiento y validación, sin fuga de información.

    θ tiene 2n valores (λ, w). Opcionalmente un valor más, v ∈ (0, 1]: la fracción de varianza explicada que deben
    cubrir las componentes que se añaden en la síntesis (variante en la que CMA-ES elige cuántas añadir). Sin v se
    añaden todas, que es el método final."""

    def __init__(self, theta, sintesis=True):
        self.theta = np.asarray(theta, float)
        self.sintesis = sintesis

    def fit(self, Xt):
        n = Xt.shape[1]
        self.lam, self.w = self.theta[:n], self.theta[n:2 * n]
        self.v = float(self.theta[2 * n]) if len(self.theta) > 2 * n else 1.0
        T = yeo_johnson(Xt, self.lam)
        self.mu, self.sd = T.mean(0), T.std(0)
        self.sd[self.sd == 0] = 1.0                                # atributos constantes: evitar /0
        Z = self._espacio(Xt)
        self.z_media = Z.mean(0)
        # Componentes principales del espacio aprendido: V = vectores singulares derechos de Z centrado
        _, S, Vt = np.linalg.svd(Z - self.z_media, full_matrices=False)
        self.V = Vt.T
        acumulada = np.cumsum(S ** 2) / max(np.sum(S ** 2), 1e-300)
        # nº de componentes que se añaden: todas (v = 1) o las primeras que cubren la fracción v de la varianza
        self.m = n if self.v >= 1.0 else int(np.searchsorted(acumulada, self.v - 1e-12) + 1)
        return self

    def _espacio(self, X):
        """Bloques 1 y 2: forma (YJ) + estandarizar + ponderar."""
        return self.w * (yeo_johnson(X, self.lam) - self.mu) / self.sd

    def transform(self, X):
        Z = self._espacio(X)
        if not self.sintesis:
            return Z
        return np.hstack([Z, (Z - self.z_media) @ self.V[:, :self.m]])   # bloque 3: añadir las componentes


# --------------------------------------------------------------------------------------------------
# Fitness: lo que CMA-ES minimiza
# --------------------------------------------------------------------------------------------------
def knn_rapido(Za, ya, Zb, k=5):
    """KNN por fuerza bruta en NumPy (equivale a KNeighborsRegressor salvo empates). Se llama miles de veces."""
    d = (Zb ** 2).sum(1)[:, None] + (Za ** 2).sum(1)[None, :] - 2 * Zb @ Za.T
    return ya[np.argpartition(d, k - 1, axis=1)[:, :k]].mean(1)


def hacer_fitness(Xs, ys, semilla, k=5, pliegues=5, sintesis=False):
    """f(θ) = MAE de validación cruzada de KNN en el espacio transformado / MAE del predictor trivial (mediana).
    f = 1 es tan malo como no saber nada; f = 0 es perfecto. Normalizar hace comparable el fitness entre datasets.
    Los pliegues se fijan una vez para que todos los candidatos se comparen en igualdad de condiciones.
    sintesis=True solo hace falta para la variante en la que θ incluye la fracción de varianza v."""
    escala = np.mean(np.abs(ys - np.median(ys)))
    plie = list(KFold(pliegues, shuffle=True, random_state=semilla).split(Xs))

    def f(theta):
        try:
            # Con todas las componentes la síntesis no cambia KNN, así que no hace falta calcularla. Solo cuando se
            # añade una parte (variante con v) cambia las distancias y entra en el fitness.
            Z = Transformacion(theta, sintesis=sintesis).fit(Xs).transform(Xs)
            e = np.mean([np.mean(np.abs(knn_rapido(Z[a], ys[a], Z[b], k) - ys[b])) for a, b in plie]) / escala
            return e if np.isfinite(e) else 1e6
        except Exception:                                          # λ extremos pueden desbordar: candidato inválido
            return 1e6
    return f


# --------------------------------------------------------------------------------------------------
# Búsqueda con CMA-ES
# --------------------------------------------------------------------------------------------------
def limites(n, lim_lambda=LIM_LAMBDA):
    lo = np.r_[np.full(n, lim_lambda[0]), np.full(n, LIM_PESO[0])]
    hi = np.r_[np.full(n, lim_lambda[1]), np.full(n, LIM_PESO[1])]
    theta0 = np.ones(2 * n)          # λ = 1 (identidad) y w = 1: la estandarización de siempre
    return lo, hi, theta0


def buscar_cma(f, lo, hi, theta0, presupuesto, semilla, sigma0=0.2):
    """CMA-ES (librería `cma`, Hansen) en el cubo [0,1]^D: cada parámetro se reescala a sus límites al evaluar,
    así un único σ0 = 0.2 (20 % del rango) tiene sentido para todos. Arranca en θ0 = estandarización, de modo que
    la búsqueda solo puede mejorar sobre la línea base en el fitness. Devuelve (θ*, f(θ*), curva del mejor f)."""
    curva = []

    def g(u):
        v = f(lo + np.clip(u, 0, 1) * (hi - lo))
        curva.append(v if not curva else min(v, curva[-1]))
        return v
    r = cma.fmin(g, (theta0 - lo) / (hi - lo), sigma0,
                 options=dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
    return lo + np.clip(r[0], 0, 1) * (hi - lo), r[1], curva


def aprender(Xt, yt, semilla=0, presupuesto=1000, n_fit=2000, sintesis=True, lim_lambda=LIM_LAMBDA):
    """El programa P del enunciado. Recibe SOLO el entrenamiento y devuelve la transformación ajustada.
    El fitness usa una submuestra de como mucho n_fit filas para que cada evaluación sea rápida."""
    n = Xt.shape[1]
    idx = np.random.default_rng(semilla).choice(len(yt), min(n_fit, len(yt)), replace=False)
    f = hacer_fitness(Xt[idx], yt[idx], semilla)
    lo, hi, theta0 = limites(n, lim_lambda)
    theta, fit, curva = buscar_cma(f, lo, hi, theta0, presupuesto, semilla)
    P = Transformacion(theta, sintesis).fit(Xt)
    P.fitness, P.fitness0, P.curva = fit, curva[0], curva
    return P
