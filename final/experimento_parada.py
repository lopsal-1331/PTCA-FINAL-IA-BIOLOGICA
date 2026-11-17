"""
experimento_parada.py — Cómo se comporta CMA-ES en la búsqueda de P: generaciones, tamaños y criterio de parada.

Repite la búsqueda del método (misma configuración que metodo.aprender) en los 12 datasets con la semilla 40 y guarda,
para cada una: dimensión D = 2n, tamaño de población λ, número de padres μ, masa efectiva de selección μ_eff,
generaciones completadas, evaluaciones usadas, el criterio que detuvo la búsqueda y en qué evaluación se encontró el
mejor θ. Sirve para describir la selección, la recombinación, el reemplazo y la parada con datos reales.

Uso:  python experimento_parada.py      (≈5–10 min con 4 núcleos)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import cma
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import metodo

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "parada.csv")


def tarea(nombre, semilla=40, presupuesto=1000):
    X, y = metodo.cargar(nombre)
    Xt, _, yt, _ = train_test_split(X, y, test_size=0.3, random_state=semilla)
    idx = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    f = metodo.hacer_fitness(Xt[idx], yt[idx], semilla)
    lo, hi, th0 = metodo.limites(X.shape[1])
    g = lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo))
    es = cma.CMAEvolutionStrategy((th0 - lo) / (hi - lo), 0.2,
                                  dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
    f_ini = g((th0 - lo) / (hi - lo))
    es.optimize(g)
    r = es.result
    return dict(dataset=nombre, D=es.N, lambda_pob=es.popsize, mu=es.sp.weights.mu, mu_eff=es.sp.weights.mueff,
                generaciones=es.countiter, evaluaciones=es.countevals, parada="|".join(es.stop().keys()),
                eval_mejor=r.evals_best, fitness_inicial=f_ini, fitness_final=r.fbest,
                sigma_final=es.sigma, active=bool(es.opts["CMA_active"]))


if __name__ == "__main__":
    filas = Parallel(n_jobs=4)(delayed(tarea)(d) for d in metodo.DATASETS)
    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    print(pd.DataFrame(filas).to_string())
