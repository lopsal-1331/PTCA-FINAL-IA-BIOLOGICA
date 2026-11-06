import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np, pandas as pd, time, warnings
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import pca_lib as P
warnings.filterwarnings("ignore")

def correr(nombre, semilla, raiz="../"):
    X, y = P.cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    sub = np.random.default_rng(semilla).choice(len(yt), min(3000, len(yt)), replace=False); filas = []
    cfg = {"Random Forest (ajustado)": (RandomForestRegressor(200, random_state=semilla),
             dict(max_features=[0.3, 0.5, 0.7, 1.0], min_samples_leaf=[1, 2, 3, 5, 10], max_depth=[None, 10, 20])),
           "Gradient Boosting (ajustado)": (HistGradientBoostingRegressor(random_state=semilla),
             dict(learning_rate=loguniform(0.02, 0.3), max_leaf_nodes=[7, 15, 31, 63], min_samples_leaf=[5, 10, 20, 40],
                  l2_regularization=loguniform(1e-3, 10), max_iter=[100, 200, 400]))}
    for m, (est, esp) in cfg.items():
        t0 = time.time()
        bs = RandomizedSearchCV(est, esp, n_iter=30, cv=3, scoring="neg_mean_absolute_error", random_state=semilla, n_jobs=1, refit=False)
        bs.fit(Xt[sub], yt[sub])
        best = est.set_params(**bs.best_params_).fit(Xt, yt)
        filas.append(dict(dataset=nombre, semilla=semilla, metodo=m, mae=float(np.mean(np.abs(best.predict(Xv) - yv))), segundos=time.time() - t0))
    return filas
