"""
cribado_posthoc.py — ¿Podemos cumplir "reducción de dimensionalidad (s < n)" con lo que ya tenemos?

No relanza CMA-ES: reutiliza los θ = (λ, w) aprendidos en final/resultados/final.csv (mismas particiones, semillas
40–44) y prueba reglas de reducción aplicadas DESPUÉS de la evolución. Todas las decisiones (qué atributos quitar,
cuántas componentes dejar) se toman solo con el entrenamiento, con el mismo fitness que la evolución (MAE de VC de
KNN normalizado por la mediana).

Representaciones (n = nº de atributos originales, s = nº de atributos de X'):
    original, estándar, P (2n), P sin síntesis (n)        referencias, como en final.csv
    sel w<0.1                 quitar atributos con w_j < 0.1 (umbral fijo, a ojo)          s ≤ n
    sel τ*                    umbral aprendido: se quitan los atributos de menor peso uno a uno y se elige el menor s < n
                              cuyo fitness no es peor que el mejor + 1 error estándar (regla 1-SE, Breiman 1984)
    pca q*                    q < n componentes principales del espacio aprendido z, q elegido con la misma regla 1-SE
    sel τ* + pca              [z_S | primeras n-1-s componentes de z_S]: síntesis rellenando hasta n-1 atributos

Uso: python cribado_posthoc.py     (≈2-3 min con 20 núcleos)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, KFold

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(AQUI), "final"))
import metodo                                                       # noqa: E402
from experimento_final import modelos                               # noqa: E402

SALIDA = os.path.join(AQUI, "resultados", "cribado_posthoc.csv")
FINAL = os.path.join(os.path.dirname(AQUI), "final", "resultados", "final.csv")


def fitness_cv(Z, y, semilla, k=5, pliegues=5):
    """Igual que metodo.hacer_fitness pero devuelve también el error estándar entre pliegues (para la regla 1-SE)."""
    escala = np.mean(np.abs(y - np.median(y)))
    e = [np.mean(np.abs(metodo.knn_rapido(Z[a], y[a], Z[b], k) - y[b])) / escala
         for a, b in KFold(pliegues, shuffle=True, random_state=semilla).split(Z)]
    return np.mean(e), np.std(e, ddof=1) / np.sqrt(pliegues)


def regla_1se(curva):
    """curva: dict tamaño -> (media, se). Devuelve el menor tamaño < n con media ≤ mejor media + su se.
    Si ninguno cumple (solo el completo está dentro), se fuerza n-1: el enunciado exige s < n."""
    mejor = min(curva, key=lambda s: curva[s][0])
    techo = curva[mejor][0] + curva[mejor][1]
    n = max(curva)
    return min([s for s in curva if s < n and curva[s][0] <= techo], default=n - 1)


class Pca:
    def fit(self, Z):
        self.m = Z.mean(0)
        self.V = np.linalg.svd(Z - self.m, full_matrices=False)[2].T
        return self

    def __call__(self, Z, q):
        return (Z - self.m) @ self.V[:, :q]


def tarea(nombre, semilla, lam, w):
    X, y = metodo.cargar(nombre)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    theta = np.r_[lam, w]
    idx = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)   # como metodo.aprender

    P = metodo.Transformacion(theta, sintesis=True).fit(Xt)
    T = metodo.Transformacion(theta, sintesis=False).fit(Xt)
    Zt, Zv = T.transform(Xt), T.transform(Xv)
    Zf, yf = T.transform(Xt[idx]), yt[idx]                          # submuestra del fitness

    # --- Umbral aprendido: quitar por orden de peso creciente, elegir s con 1-SE
    orden = np.argsort(-w)                                          # de mayor a menor peso
    curva_sel = {s: fitness_cv(Zf[:, orden[:s]], yf, semilla) for s in range(1, n + 1)}
    s_sel = regla_1se(curva_sel)
    S = np.sort(orden[:s_sel])
    tau = float(w[orden[s_sel - 1]] + w[orden[s_sel]]) / 2         # umbral: entre el último que entra y el primero que sale

    # --- PCA truncada del espacio aprendido, q con 1-SE
    pz = Pca().fit(Zt)
    curva_pca = {q: fitness_cv(pz(Zf, q), yf, semilla) for q in range(1, n + 1)}
    q_pca = regla_1se(curva_pca)

    # --- Selección + síntesis hasta n-1
    ps = Pca().fit(Zt[:, S])
    m = n - 1 - s_sel

    mu, sd = Xt.mean(0), Xt.std(0)
    sd[sd == 0] = 1
    fijo = np.where(w >= 0.1)[0]
    reps = {"original": (Xt, Xv),
            "estándar": ((Xt - mu) / sd, (Xv - mu) / sd),
            "P sin síntesis": (Zt, Zv),
            "P": (P.transform(Xt), P.transform(Xv)),
            "sel w<0.1": (Zt[:, fijo], Zv[:, fijo]),
            "sel τ*": (Zt[:, S], Zv[:, S]),
            "pca q*": (pz(Zt, q_pca), pz(Zv, q_pca)),
            "sel τ* + pca": (np.hstack([Zt[:, S], ps(Zt[:, S], m)]), np.hstack([Zv[:, S], ps(Zv[:, S], m)]))}

    comun = dict(dataset=nombre, semilla=semilla, n=n, s_sel=s_sel, tau=tau, q_pca=q_pca,
                 fit_completo=curva_sel[n][0], fit_sel=curva_sel[s_sel][0], fit_pca=curva_pca[q_pca][0])
    filas = []
    for rep, (A, B) in reps.items():
        for mod, mdl in modelos(semilla).items():
            mae = float(np.mean(np.abs(mdl.fit(A, yt).predict(B) - yv)))
            filas.append(dict(comun, representacion=rep, dim=A.shape[1], modelo=mod, mae=mae))
    return filas


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    f = pd.read_csv(FINAL).drop_duplicates(["dataset", "semilla"])
    trabajo = [(r.dataset, int(r.semilla), np.array(r.lambdas.split("|"), float), np.array(r.pesos.split("|"), float))
               for r in f.itertuples()]
    filas = [x for fs in Parallel(n_jobs=int(os.environ.get("N_JOBS", os.cpu_count())))(delayed(tarea)(*t) for t in trabajo) for x in fs]
    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    print("TERMINADO", len(filas), "filas ->", SALIDA)
