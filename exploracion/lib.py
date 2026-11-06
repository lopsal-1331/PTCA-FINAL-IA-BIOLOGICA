"""Test rápido de técnicas bioinspiradas (Tabla 1 de Fister et al. 2013) para aprender la transformación."""
import numpy as np, pandas as pd, time, warnings, importlib
from scipy.stats import yeojohnson
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings("ignore")

RUTAS = {"diabetes": ("DATASETS/Diabetes.csv", "target"), "california": ("DATASETS/California.csv", "MedHouseVal")}
def cargar(nombre, raiz="../"):
    f, t = RUTAS[nombre]; df = pd.read_csv(raiz + f)
    return df.drop(columns=t).values.astype(float), df[t].values.astype(float)

def ajustar(Xt, theta):
    n = Xt.shape[1]; lam, w = theta[:n], theta[n:]
    C = np.column_stack([yeojohnson(Xt[:, j], lmbda=lam[j]) for j in range(n)])
    mu, sd = C.mean(0), C.std(0); sd[sd == 0] = 1
    def v(X):
        Z = np.column_stack([yeojohnson(X[:, j], lmbda=lam[j]) for j in range(n)])
        return w * (Z - mu) / sd
    return v

def knn5(Za, ya, Zb):
    d = (Zb ** 2).sum(1)[:, None] + (Za ** 2).sum(1)[None, :] - 2 * Zb @ Za.T
    return ya[np.argpartition(d, 4, axis=1)[:, :5]].mean(1)

def knn_k(Za, ya, Zb, k):
    d = (Zb ** 2).sum(1)[:, None] + (Za ** 2).sum(1)[None, :] - 2 * Zb @ Za.T
    return ya[np.argpartition(d, k - 1, axis=1)[:, :k]].mean(1)

def lineal(Za, ya, Zb):
    b = np.linalg.lstsq(np.c_[np.ones(len(Za)), Za], ya, rcond=None)[0]
    return np.c_[np.ones(len(Zb)), Zb] @ b

def hacer_fitness(Xs, ys, semilla):
    """Igual que el notebook final: media de MAE normalizado de Regresión Lineal y KNN, CV 5 pliegues."""
    escala = np.mean(np.abs(ys - np.median(ys)))
    plie = list(KFold(5, shuffle=True, random_state=semilla).split(Xs))
    cuenta = [0]
    def f(theta):
        cuenta[0] += 1
        try:
            Z = ajustar(Xs, np.asarray(theta, float))(Xs); tot = 0.0
            for a, b in plie:
                tot += np.mean(np.abs(lineal(Z[a], ys[a], Z[b]) - ys[b]))
                tot += np.mean(np.abs(knn5(Z[a], ys[a], Z[b]) - ys[b]))
            r = tot / 10 / escala
            return r if np.isfinite(r) else 1e6
        except Exception:
            return 1e6
    return f, cuenta

def evaluar(Zt, Zv, yt, yv):
    m = {"Regresión Lineal": LinearRegression(), "KNN": KNeighborsRegressor(), "Árbol de decisión": DecisionTreeRegressor(random_state=13)}
    return {k: float(np.mean(np.abs(mod.fit(Zt, yt).predict(Zv) - yv))) for k, mod in m.items()}

# ---------------- algoritmos de la Tabla 1 (mealpy) + referencias
ALGS = {  # etiqueta : (módulo mealpy, familia en el paper)
    "PSO": ("PSO", "enjambre"), "ABC": ("ABC", "enjambre"), "ACOR": ("ACOR", "enjambre"),
    "BA (murciélagos)": ("BA", "enjambre"), "BFO (bacterias)": ("BFO", "enjambre"),
    "CSO (gatos)": ("CSO", "enjambre"), "FFA (luciérnagas)": ("FFA", "enjambre"),
    "CSA (cuco)": ("CSA", "enjambre"),
    "FPA (polinización)": ("FPA", "bio-no-SI"), "BBO (biogeografía)": ("BBO", "bio-no-SI"),
    "DE": ("DE", "bio-no-SI"), "IWO (malezas)": ("IWO", "bio-no-SI"), "BSO (tormenta ideas)": ("BSO", "bio-no-SI"),
    "GA (propio)": ("GA", "evolutivo"), "ES": ("ES", "evolutivo"), "EP": ("EP", "evolutivo"),
    "SA": ("SA", "física"), "HS (armonía)": ("HS", "física"), "WCA (ciclo agua)": ("WCA", "física"),
    "ICA (imperialista)": ("ICA", "otros"), "BSA (backtracking)": ("BSA", "otros"),
}
def clase(modulo):
    from mealpy import Optimizer
    mod = getattr(importlib.import_module("mealpy"), modulo)
    for n in (f"Original{modulo}", f"Base{modulo}"):
        if hasattr(mod, n): return getattr(mod, n)
    for n in dir(mod):
        c = getattr(mod, n)
        if isinstance(c, type) and issubclass(c, Optimizer) and c is not Optimizer: return c

def correr(nombre, semilla, alg, raiz="../", presupuesto=1000, pop=30):
    X, y = cargar(nombre, raiz)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=semilla)
    n = X.shape[1]
    sub = np.random.default_rng(semilla).choice(len(yt), min(2000, len(yt)), replace=False)
    f, cuenta = hacer_fitness(Xt[sub], yt[sub], semilla)
    lo = np.r_[np.full(n, -2.0), np.zeros(n)]; hi = np.r_[np.full(n, 2.0), np.full(n, 3.0)]
    th0 = np.ones(2 * n); t0 = time.time()
    if alg == "CMA-ES":
        import cma
        g = lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo))
        r = cma.fmin(g, (th0 - lo) / (hi - lo), 0.2, options=dict(seed=semilla + 1, bounds=[0, 1], maxfevals=presupuesto, verbose=-9))
        th = lo + np.clip(r[0], 0, 1) * (hi - lo)
    elif alg == "búsqueda aleatoria":
        rng = np.random.default_rng(semilla); mejor, th = f(th0), th0
        for _ in range(presupuesto - 1):
            c = rng.uniform(lo, hi); v = f(c)
            if v < mejor: mejor, th = v, c
    elif alg == "GA (propio)":
        rng = np.random.default_rng(semilla); N = 30; D = len(lo)
        P = np.vstack([th0, rng.uniform(lo, hi, (N - 1, D))]); F = np.array([f(p) for p in P])
        while cuenta[0] < presupuesto:
            hijos = []
            while len(hijos) < N - 1 and cuenta[0] + len(hijos) < presupuesto:
                i = [min(rng.integers(0, N, 3), key=lambda j: F[j]) for _ in range(2)]
                a, b = P[i[0]], P[i[1]]
                c = a + rng.uniform(-0.3, 1.3, D) * (b - a)                  # cruce BLX-α
                m = rng.random(D) < 1.0 / D
                c = np.where(m, c + rng.normal(0, 0.1, D) * (hi - lo), c)   # mutación gaussiana
                hijos.append(np.clip(c, lo, hi))
            Fh = np.array([f(h) for h in hijos])
            elite = np.argmin(F)
            P = np.vstack([P[elite], hijos]); F = np.r_[F[elite], Fh]; N = len(P)
        th = P[np.argmin(F)]
    elif alg == "θ inicial (sin búsqueda)":
        th = th0
    else:
        from mealpy import FloatVar
        rng = np.random.default_rng(semilla)
        ini = [th0] + [rng.uniform(lo, hi) for _ in range(pop - 1)]
        mejor = {"v": np.inf, "x": th0}
        class Tope(Exception): pass
        def obj(x):
            if cuenta[0] >= presupuesto: raise Tope()      # tope duro de evaluaciones
            v = f(x)
            if v < mejor["v"]: mejor["v"], mejor["x"] = v, np.array(x, float)
            return v
        prob = {"obj_func": obj, "bounds": FloatVar(lb=lo, ub=hi), "minmax": "min", "log_to": None}
        mod = clase(ALGS[alg][0])
        try: m = mod(epoch=100000, pop_size=pop)
        except TypeError: m = mod(epoch=100000)
        try:
            try: m.solve(prob, termination={"max_fe": presupuesto}, seed=semilla, starting_solutions=ini)
            except TypeError: m.solve(prob, termination={"max_fe": presupuesto}, seed=semilla)
        except Tope:
            pass
        th = mejor["x"]
    seg = time.time() - t0
    v = ajustar(Xt, th)
    d = dict(dataset=nombre, semilla=semilla, algoritmo=alg, fitness=f(th), evaluaciones=cuenta[0], segundos=seg)
    d.update(evaluar(v(Xt), v(Xv), yt, yv))
    ref = evaluar(Xt, Xv, yt, yv)
    d.update({"crudo " + k: val for k, val in ref.items()})
    return d
