import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np, pandas as pd, time, warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import pca_lib as P
warnings.filterwarnings("ignore")

def correr(nombre, semilla, raiz="../", n_gp=2000):
    X, y = P.cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]; sub = np.random.default_rng(semilla).choice(len(yt), min(n_gp, len(yt)), replace=False)
    mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1; ym, ys_ = yt.mean(), yt.std()
    filas = []
    def reg(m, t0, mae, extra=None): filas.append(dict(dataset=nombre, semilla=semilla, metodo=m, mae=float(mae), segundos=time.time() - t0))
    t0 = time.time()
    rf = RandomForestRegressor(200, n_jobs=1, random_state=semilla).fit(Xt, yt); reg("Random Forest", t0, np.mean(np.abs(rf.predict(Xv) - yv)), None)
    t0 = time.time()
    hg = HistGradientBoostingRegressor(random_state=semilla).fit(Xt, yt); reg("Gradient Boosting", t0, np.mean(np.abs(hg.predict(Xv) - yv)))
    t0 = time.time()
    k = C(1.0) * RBF(length_scale=np.full(n, 2.0), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(0.1, (1e-5, 1))
    gp = GaussianProcessRegressor(k, normalize_y=False, random_state=semilla, n_restarts_optimizer=0)
    Zs = (Xt[sub] - mu) / sd; gp.fit(Zs, (yt[sub] - ym) / ys_)
    reg("Proceso gaussiano ARD (n=2000)", t0, np.mean(np.abs(gp.predict((Xv - mu) / sd) * ys_ + ym - yv)))
    return filas
if __name__ == "__main__":
    import sys; t=time.time(); fs=correr("diabetes",30); print([(f["metodo"],round(f["mae"],2),round(f["segundos"])) for f in fs], round(time.time()-t))
