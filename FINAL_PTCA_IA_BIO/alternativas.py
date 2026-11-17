"""
alternativas.py — Transformaciones con las que se compara P (todas aprendidas solo con el entrenamiento).

estándar, Yeo-Johnson por máxima verosimilitud, cuantiles, "solo PCA" (95 % de varianza o % elegido por validación
cruzada de KNN), X estandarizado + su PCA, MLKR (aprendizaje de métricas para regresión) y construcción de atributos
con programación genética (gplearn). MLKR y la programación genética usan una submuestra de 1000 filas (su coste es
cuadrático / poblacional).
"""
import functools
import sys
import numpy as np
import sklearn.utils as U
import sklearn.utils.validation as V
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
import ptca


def compatibilidad():
    """metric-learn y gplearn llaman a check_X_y/check_array con `force_all_finite`, que scikit-learn >= 1.6 renombró
    a `ensure_all_finite`: se traduce el argumento. Idempotente; se llama en cada proceso."""
    def traducir(f):
        @functools.wraps(f)
        def g(*args, **kwargs):
            if "force_all_finite" in kwargs:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
            return f(*args, **kwargs)
        g._compat = True
        return g
    for nombre in ["check_X_y", "check_array"]:
        f = getattr(V, nombre)
        if not getattr(f, "_compat", False):
            f = traducir(f)
            setattr(V, nombre, f)
        setattr(U, nombre, f)
        for mod in list(sys.modules.values()):
            if mod is not None and getattr(mod, "__name__", "").split(".")[0] in ("metric_learn", "gplearn") \
                    and hasattr(mod, nombre):
                setattr(mod, nombre, f)


def _pca(Z):
    m = Z.mean(0)
    _, S, Vt = np.linalg.svd(Z - m, full_matrices=False)
    return m, Vt.T, np.cumsum(S ** 2) / np.sum(S ** 2)


def _v_por_cv(Zt, yt, semilla):
    idx = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    Zs, ys = Zt[idx], yt[idx]
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Zs))
    mejor = (np.inf, 1.0)
    for v in np.arange(0.5, 1.0001, 0.05):
        err = []
        for a, b in plie:
            m, Vc, c = _pca(Zs[a])
            k = int(np.searchsorted(c, v - 1e-9) + 1)
            err.append(np.mean(np.abs(ptca._knn((Zs[a] - m) @ Vc[:, :k], ys[a], (Zs[b] - m) @ Vc[:, :k]) - ys[b])))
        if np.mean(err) < mejor[0]:
            mejor = (np.mean(err), v)
    return mejor[1]


def representaciones(Xt, Xv, yt, semilla):
    """nombre -> función que ajusta con Xt y devuelve (A, B) = (X'_t, X'_v)."""
    def estandar():
        s = StandardScaler().fit(Xt)
        return s.transform(Xt), s.transform(Xv)

    def yj_mv():
        p = PowerTransformer(method="yeo-johnson", standardize=True).fit(Xt)
        return p.transform(Xt), p.transform(Xv)

    def cuantiles():
        q = QuantileTransformer(n_quantiles=min(1000, len(Xt)), output_distribution="normal",
                                random_state=semilla).fit(Xt)
        return q.transform(Xt), q.transform(Xv)

    def pca_v(v):
        def f():
            A, B = estandar()
            m, Vc, c = _pca(A)
            k = int(np.searchsorted(c, (_v_por_cv(A, yt, semilla) if v is None else v) - 1e-9) + 1)
            return (A - m) @ Vc[:, :k], (B - m) @ Vc[:, :k]
        return f

    def est_pca():
        A, B = estandar()
        m, Vc, _ = _pca(A)
        return np.hstack([A, (A - m) @ Vc]), np.hstack([B, (B - m) @ Vc])

    def mlkr():
        compatibilidad()
        from metric_learn import MLKR
        compatibilidad()
        A, B = estandar()
        idx = np.random.default_rng(semilla).choice(len(yt), min(1000, len(yt)), replace=False)
        L = MLKR(max_iter=100, random_state=semilla).fit(A[idx], yt[idx])
        return L.transform(A), L.transform(B)

    def pg():
        compatibilidad()
        from gplearn.genetic import SymbolicTransformer
        compatibilidad()
        A, B = estandar()
        idx = np.random.default_rng(semilla).choice(len(yt), min(1000, len(yt)), replace=False)
        g = SymbolicTransformer(population_size=500, generations=10, n_components=A.shape[1], hall_of_fame=50,
                                function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"),
                                parsimony_coefficient=0.001, random_state=semilla, n_jobs=1).fit(A[idx], yt[idx])
        Ga, Gb = g.transform(A), g.transform(B)
        s = StandardScaler().fit(Ga)
        return np.hstack([A, np.nan_to_num(s.transform(Ga))]), np.hstack([B, np.nan_to_num(s.transform(Gb))])

    return {"estándar": estandar, "YJ máx. verosimilitud": yj_mv, "cuantiles": cuantiles, "PCA 95 %": pca_v(0.95),
            "PCA v-CV": pca_v(None), "estándar + PCA": est_pca, "MLKR": mlkr, "PG (gplearn)": pg}
