"""
analisis.py — Tablas, contrastes y figuras a partir de los CSV de resultados/.

Es la ÚNICA fuente de números y gráficas del notebook, la memoria y el póster. `python analisis.py` regenera las
figuras (figuras/*.pdf|png), las cifras (numeros.tex) y las filas de las tablas (tablas.tex).

Convenciones
------------
* Se compara el MAE de validación de una configuración con el de una referencia mediante el cociente MAE/MAE_ref
  (< 1 = mejora). Dentro de cada dataset se promedian las particiones con la MEDIA GEOMÉTRICA, y entre datasets se
  vuelve a usar la media geométrica (los datasets tienen escalas muy distintas y los cocientes no son simétricos).
* Estadística POR DATASET (Demšar, 2006): la unidad de análisis es el dataset (n = 12), no la partición, porque las
  particiones de un mismo dataset no son independientes (comparten filas). Comparaciones por pares: Wilcoxon de
  rangos con signo sobre los 12 cocientes (en escala log), con corrección de Holm dentro de cada familia de
  contrastes. Comparaciones múltiples: test de Friedman sobre los rangos por dataset y diferencia crítica de Nemenyi.
* También se informa de cuántos casos (dataset × partición) mejoran, como medida descriptiva de consistencia.

Fuentes
-------
final.csv        confirmación de P sin reducir (2n atributos): 12 datasets × 5 particiones (semillas 40–44)
../reduccion/resultados/reduccion.csv
                 P final (con reducción aprendida por CMA-ES, < n atributos) en esas mismas particiones:
                 'τ+ρ α=0' = P; 'τ α=0' = P sin síntesis (solo reducción)
comparativa.csv  alternativas de preprocesado / representación en esas mismas particiones
busqueda.csv     variantes de la búsqueda con la configuración final (varianza, controles, rango, algoritmos)
lambdas_mle.csv  λ de máxima verosimilitud de cada atributo
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import wilcoxon, friedmanchisquare, studentized_range

CARPETA = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CARPETA, "resultados")
RED = os.path.join(os.path.dirname(CARPETA), "reduccion", "resultados", "reduccion.csv")
VAR_RED = {"τ+ρ α=0": "P", "τ α=0": "P sin síntesis"}      # variantes de reduccion.csv -> nombre en el análisis
FIG = os.path.join(CARPETA, "figuras")

# ---------------------------------------------------------------- estilo
# Paleta categórica validada (claro, CVD): azul = método, naranja = alternativas, verde = extra; gris = referencias.
AZUL, NARANJA, VERDE, GRIS, TINTA, TINTA2 = "#2E6DB4", "#D9731A", "#2F9467", "#A3A9AF", "#1D2329", "#5A6772"
DIVERGENTE = LinearSegmentedColormap.from_list("div", [AZUL, "#F2F2F0", NARANJA])   # mejora – igual – empeora
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#C9CED3", "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ECEEF0",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 110, "savefig.bbox": "tight",
    "legend.frameon": False, "text.color": TINTA})

NOMBRES = {"concrete_compressive_strength": "concrete"}
MODELOS = ["Regresión lineal", "KNN", "Árbol", "Random Forest"]
CORTO_MOD = {"Regresión lineal": "R. lineal", "KNN": "KNN", "Árbol": "Árbol", "Random Forest": "RF"}


def corto(d):
    return NOMBRES.get(d, d)


def geo(s):
    s = np.asarray(s, float)
    return float(np.exp(np.log(s).mean()))


def guardar(fig, nombre):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, nombre + ".pdf"))
    fig.savefig(os.path.join(FIG, nombre + ".png"), dpi=200)


# ================================================================ estadística por dataset
def holm(p):
    """p-valores ajustados por Holm (controla el error de familia)."""
    p = np.asarray(p, float)
    orden = np.argsort(p)
    aj = np.empty_like(p)
    acumulado = 0.0
    for i, j in enumerate(orden):
        acumulado = max(acumulado, (len(p) - i) * p[j])
        aj[j] = min(1.0, acumulado)
    return aj


def resumen(casos):
    """casos: DataFrame con columnas dataset, semilla, rel (= MAE/MAE_ref). Devuelve el cociente medio, los casos y
    datasets que mejoran y el p de Wilcoxon por dataset (n = nº de datasets)."""
    por_ds = casos.groupby("dataset").rel.apply(geo)
    lg = np.log(por_ds.values)
    p = wilcoxon(lg).pvalue if np.any(np.abs(lg) > 1e-12) else 1.0
    g = geo(por_ds)
    return dict(cociente=g, cambio=100 * (g - 1), casos_mejora=int((casos.rel < 1).sum()), casos=len(casos),
                ds_mejora=int((por_ds < 1).sum()), ds=len(por_ds), p=float(p), peor_ds=float(por_ds.max()))


def friedman_nemenyi(tabla):
    """tabla: datasets × métodos (MAE o cocientes; menor = mejor). Rango medio por método, p de Friedman y
    diferencia crítica de Nemenyi (α = 0,05)."""
    rangos = tabla.rank(axis=1).mean()
    N, k = tabla.shape
    p = friedmanchisquare(*[tabla[c].values for c in tabla.columns]).pvalue
    q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    return rangos.sort_values(), float(p), float(q * np.sqrt(k * (k + 1) / (6 * N)))


def diagrama_cd(ax, rangos, cd, titulo):
    """Diagrama de diferencia crítica (Demšar, 2006): rangos medios en un eje; las barras gruesas unen métodos
    cuya diferencia de rango es menor que la diferencia crítica (no distinguibles)."""
    k = len(rangos)
    nombres, r = list(rangos.index), rangos.values
    ax.set_xlim(k + 0.3, 0.7)                     # rango 1 (mejor) a la derecha
    ax.set_ylim(0, 1.08)
    ax.axis("off")
    ax.hlines(0.86, 1, k, color=TINTA2, lw=1)
    for i in range(1, k + 1):
        ax.vlines(i, 0.86, 0.9, color=TINTA2, lw=1)
        ax.text(i, 0.93, str(i), ha="center", fontsize=7, color=TINTA2)
    ax.hlines(1.02, 1, 1 + cd, color=TINTA, lw=1.5)
    ax.text(1 + cd / 2, 1.04, f"CD = {cd:.2f}".replace(".", ","), ha="center", fontsize=7)
    mitad = (k + 1) // 2
    for i, (nom, ri) in enumerate(zip(nombres, r)):              # los mejores, a la derecha
        derecha = i < mitad
        y = 0.72 - 0.11 * (i if derecha else i - mitad)
        x_txt = 0.75 if derecha else k + 0.25
        col = AZUL if nom in ("P", "P sin síntesis", "P sin reducir") else TINTA
        ax.plot([ri, ri, x_txt], [0.86, y, y], color=col, lw=0.9)
        ax.text(x_txt + (-0.05 if derecha else 0.05), y, f"{nom} ({ri:.1f})".replace(".", ","),
                ha="left" if derecha else "right", va="center", fontsize=7, color=col)
    # cliques: tramos maximales de métodos consecutivos (por rango) con diferencia < CD
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for n_g, (a, b) in enumerate(grupos):
        y = 0.82 - 0.025 * n_g
        ax.hlines(y, r[a] - 0.03, r[b] + 0.03, color=TINTA, lw=2.5)
    ax.set_title(titulo, fontsize=9, loc="left")


# ================================================================ 1. Confirmación del método
def cargar_reduccion():
    """Filas de P final y P sin síntesis (reduccion.csv) con el formato de final.csv."""
    r = pd.read_csv(RED)
    r = r[r.variante.isin(VAR_RED)].copy()
    r["representacion"] = r.variante.map(VAR_RED)
    return r.drop(columns="variante")


def cargar_final():
    """final.csv (original, estándar y el P anterior, que pasa a llamarse 'P sin reducir') + P final de
    reduccion.csv. Se descarta el antiguo 'P sin síntesis' (bloques 1–2 sin reducir)."""
    f = pd.read_csv(os.path.join(RES, "final.csv"))
    f = f[f.representacion != "P sin síntesis"].replace({"representacion": {"P": "P sin reducir"}})
    f["dim"] = np.where(f.representacion == "P sin reducir", 2 * f.n_atributos, f.n_atributos)
    r = cargar_reduccion()
    info = f.drop_duplicates(["dataset", "semilla"])[["dataset", "semilla", "n_filas", "n_atributos"]]
    r = r.drop(columns=[c for c in ["n_filas", "n_atributos"] if c in r]).merge(info, on=["dataset", "semilla"])
    d = pd.concat([f, r], ignore_index=True)
    clave = list(zip(d.dataset, d.semilla, d.modelo))
    for ref, col in [("original", "rel"), ("estándar", "rel_est")]:
        r = d[d.representacion == ref].set_index(["dataset", "semilla", "modelo"]).mae
        d[col] = d.mae.values / r.loc[clave].values
    return d


def tabla_principal(d):
    """Por modelo: e'/e de P frente a original y a estándar. Holm sobre los 8 contrastes (4 modelos × 2 refs)."""
    filas = []
    for m in MODELOS:
        s = d[(d.modelo == m) & (d.representacion == "P")]
        a = resumen(s)
        b = resumen(s.assign(rel=s.rel_est))
        sin = d[(d.modelo == m) & (d.representacion == "P sin síntesis")]
        sr = d[(d.modelo == m) & (d.representacion == "P sin reducir")]
        est = d[(d.modelo == m) & (d.representacion == "estándar")]
        filas.append(dict(modelo=m, estandar=geo(est.groupby("dataset").rel.apply(geo)),
                          sin_sintesis=geo(sin.groupby("dataset").rel.apply(geo)),
                          sin_reducir=geo(sr.groupby("dataset").rel.apply(geo)),
                          **{k + "_orig": v for k, v in a.items()}, **{k + "_est": v for k, v in b.items()}))
    t = pd.DataFrame(filas).set_index("modelo")
    aj = holm(np.r_[t.p_orig.values, t.p_est.values])
    t["p_holm_orig"], t["p_holm_est"] = aj[:4], aj[4:]
    return t


def tabla_por_dataset(d, rep="P"):
    s = d[d.representacion == rep]
    t = s.groupby(["dataset", "modelo"]).rel.apply(geo).unstack()[MODELOS]
    return t.loc[list(d.dataset.unique())]


def efecto(d, rep="P", ref="P sin síntesis"):
    """Cociente MAE(rep)/MAE(ref) por modelo (contraste por dataset, Holm sobre los 4 modelos)."""
    filas = []
    for m in MODELOS:
        s = d[d.modelo == m]
        con = s[s.representacion == rep].set_index(["dataset", "semilla"]).mae
        sin = s[s.representacion == ref].set_index(["dataset", "semilla"]).mae
        com = con.index.intersection(sin.index)
        filas.append(dict(modelo=m, **resumen((con[com] / sin[com]).rename("rel").reset_index())))
    t = pd.DataFrame(filas).set_index("modelo")
    t["p_holm"] = holm(t.p.values)
    return t


def efecto_sintesis(d):
    """Qué aporta la síntesis dentro del presupuesto de n-1 atributos: P frente a P sin síntesis (solo reducción)."""
    return efecto(d, "P", "P sin síntesis")


def efecto_reduccion(d):
    """Qué cuesta reducir: P (< n atributos) frente a P sin reducir (2n atributos)."""
    return efecto(d, "P", "P sin reducir")


def tabla_dimension(d):
    """Por dataset: n, atributos que se quedan (s), componentes añadidas (m), dim(X') y umbral τ aprendido
    (media de las particiones)."""
    u = d[d.representacion == "P"].drop_duplicates(["dataset", "semilla"])
    t = u.groupby("dataset").agg(n=("n_atributos", "first"), s=("s", "mean"), m=("m", "mean"), dim=("dim", "mean"),
                                 tau=("tau", "median"), rho=("rho", "median"))
    t["dim/n"] = t.dim / t.n
    return t.loc[[x for x in d.dataset.unique() if x in t.index]]


def _barras(a1, d, titulo=True, fs=8):
    reps = [("estándar", GRIS), ("P sin reducir", NARANJA), ("P", AZUL)]
    x = np.arange(len(MODELOS))
    w = 0.26
    for i, (rep, c) in enumerate(reps):
        vals = [geo(d[(d.modelo == m) & (d.representacion == rep)].groupby("dataset").rel.apply(geo))
                for m in MODELOS]
        a1.bar(x + (i - 1) * w, np.array(vals) - 1, w * 0.92, bottom=1, color=c,
               label={"P sin reducir": "P sin reducir (2n atrib.)", "P": "P (< n atrib.)"}.get(rep, rep))
        if rep == "P":
            for xi, v in zip(x, vals):
                a1.text(xi + w, v - 0.012 if v < 1 else v + 0.004, f"{100 * (v - 1):+.1f} %".replace(".", ","),
                        ha="center", va="top" if v < 1 else "bottom", fontsize=fs, color=TINTA)
    a1.axhline(1, color=TINTA2, lw=1)
    a1.set_xticks(x, ["Regresión\nlineal", "KNN", "Árbol", "Random\nForest"])
    a1.set_ylabel("MAE / MAE con X original  (<1 mejora)")
    a1.set_ylim(min(0.7, a1.get_ylim()[0]), 1.04)
    a1.legend(loc="lower left", fontsize=fs)
    if titulo:
        a1.set_title("e'/e por modelo (media geométrica de los 12 datasets)", loc="left", fontsize=9)


def _heatmap(a2, d, titulo=True, fs=7.5):
    t = tabla_por_dataset(d)
    t = t.loc[t.mean(axis=1).sort_values().index]
    lim = max(abs(np.log(t.values)).max(), 0.05)
    a2.imshow(np.log(t.values), cmap=DIVERGENTE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    a2.set_xticks(range(4), ["R. lineal", "KNN", "Árbol", "RF"])
    a2.set_yticks(range(len(t)), [corto(i) for i in t.index])
    a2.grid(False)
    for i in range(t.shape[0]):
        for j in range(4):
            v = t.values[i, j]
            a2.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=fs,
                    color="white" if abs(np.log(v)) > 0.6 * lim else TINTA)
    if titulo:
        a2.set_title("e'/e del método P por dataset", loc="left", fontsize=9)


def fig_principal(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.05, 1]))
    _barras(a1, d)
    _heatmap(a2, d)
    fig.tight_layout()
    return fig


def figs_poster(d):
    f1, a = plt.subplots(figsize=(4.6, 3.3))
    _barras(a, d, titulo=False, fs=8)
    f1.tight_layout()
    f2, a = plt.subplots(figsize=(3.9, 3.9))
    _heatmap(a, d, titulo=False, fs=7.5)
    f2.tight_layout()
    return f1, f2


# ================================================================ 2. Comparativa con alternativas
ORDEN_COMP = ["estándar", "YJ máx. verosimilitud", "cuantiles", "PCA 95 %", "PCA v-CV", "estándar + PCA",
              "MLKR", "PG (gplearn)", "P sin reducir", "P"]


def cargar_comparativa():
    """comparativa.csv (el P anterior pasa a 'P sin reducir'; se descarta su 'P sin síntesis') + P final y P sin
    síntesis de reduccion.csv, en formato largo, con rel frente a original y n_atributos_salida."""
    c = pd.read_csv(os.path.join(RES, "comparativa.csv"))
    c = c[c.representacion != "P sin síntesis"].replace({"representacion": {"P": "P sin reducir"}})
    r = cargar_reduccion().rename(columns={"dim": "n_atributos_salida", "segundos": "segundos_transf"})
    c = pd.concat([c, r[["dataset", "semilla", "representacion", "modelo", "mae", "segundos_transf",
                         "n_atributos_salida"]]], ignore_index=True)
    n = c[c.representacion == "original"].drop_duplicates(["dataset", "semilla"]).set_index(["dataset", "semilla"])
    c["frac_dim"] = c.n_atributos_salida.values / n.n_atributos_salida.loc[list(zip(c.dataset, c.semilla))].values
    ref = c[c.representacion == "original"].set_index(["dataset", "semilla", "modelo"]).mae
    c = c[c.set_index(["dataset", "semilla", "modelo"]).index.isin(ref.index)].copy()
    c["rel"] = c.mae.values / ref.loc[list(zip(c.dataset, c.semilla, c.modelo))].values
    return c


def _completos(c, reps=None):
    """Solo particiones donde están TODAS las representaciones (comparación pareada)."""
    if reps is not None:
        c = c[c.representacion.isin(reps)]
    n = c.groupby(["dataset", "semilla"]).representacion.nunique()
    idx = n[n == c.representacion.nunique()].index
    return c[c.set_index(["dataset", "semilla"]).index.isin(idx)], len(idx)


def tabla_comparativa(c):
    """Cociente frente a original (media geométrica entre datasets) de cada representación y modelo, rango medio de
    Friedman y p de Wilcoxon por dataset de P frente a cada alternativa (Holm sobre las alternativas)."""
    c, n = _completos(c, ORDEN_COMP + ["original"])
    reps = [r for r in ORDEN_COMP if r in c.representacion.unique()]
    t = pd.DataFrame(index=reps)
    for m in MODELOS:
        por_ds = c[c.modelo == m].groupby(["dataset", "representacion"]).rel.apply(geo).unstack()[reps]
        t[m] = por_ds.apply(geo)
        rangos, _, _ = friedman_nemenyi(por_ds)
        t["rango " + m] = rangos
        ps = []
        for r in reps:
            x = np.log(por_ds["P"] / por_ds[r])
            ps.append(np.nan if r == "P" else (wilcoxon(x).pvalue if np.any(np.abs(x) > 1e-12) else 1.0))
        ps = np.array(ps)
        aj = np.full_like(ps, np.nan)
        aj[~np.isnan(ps)] = holm(ps[~np.isnan(ps)])
        t["p " + m] = aj
    t["rango medio"] = t[[f"rango {m}" for m in MODELOS]].mean(axis=1)
    t["dim/n"] = c.drop_duplicates(["dataset", "semilla", "representacion"]).groupby("representacion").frac_dim.mean()
    return t, n


def fig_comparativa_cd(c, modelos=MODELOS, figsize=(10.5, 5.8), columnas=2):
    c, _ = _completos(c, ORDEN_COMP + ["original"])
    reps = [r for r in ORDEN_COMP if r in c.representacion.unique()]
    columnas = min(columnas, len(modelos))
    filas = (len(modelos) + columnas - 1) // columnas
    fig, ejes = plt.subplots(filas, columnas, figsize=figsize)
    for ax, m in zip(np.atleast_1d(ejes).ravel(), modelos):
        por_ds = c[c.modelo == m].groupby(["dataset", "representacion"]).rel.apply(geo).unstack()[reps]
        rangos, p, cd = friedman_nemenyi(por_ds)
        diagrama_cd(ax, rangos, cd, f"{m}  (Friedman p = {p:.1e})".replace(".", ","))
    fig.tight_layout()
    return fig


# ================================================================ 3. Diseño de la síntesis (árboles)
SINT = ["PCA 95 %", "PCA v-CV", "estándar + PCA", "P sin reducir", "P sin síntesis", "P"]
SINT_TXT = {"PCA 95 %": "solo PCA (95 % varianza)", "PCA v-CV": "solo PCA (% por VC)",
            "estándar + PCA": "X estándar + su PCA (2n)", "P sin reducir": "P sin reducir (2n)",
            "P sin síntesis": "P solo reducción (< n)", "P": "P reducción + síntesis (< n)"}


def fig_sintesis(c, modelos=("Árbol", "Random Forest"), figsize=(9.5, 3.3)):
    """Mismas particiones para todas las variantes: punto = dataset (media de particiones), barra = media
    geométrica entre datasets; frente a X original."""
    c, _ = _completos(c, SINT + ["original"])
    reps = [r for r in SINT if r in c.representacion.unique()]
    fig, ejes = plt.subplots(1, len(modelos), figsize=figsize, sharey=True)
    rng = np.random.default_rng(0)
    for ax, m in zip(np.atleast_1d(ejes), modelos):
        por_ds = c[c.modelo == m].groupby(["dataset", "representacion"]).rel.apply(geo).unstack()
        for i, r in enumerate(reps):
            v = por_ds[r].dropna()
            g = geo(v)
            col = AZUL if r == "P" else (NARANJA if r in ("P sin reducir", "P sin síntesis") else GRIS)
            ax.barh(i, g - 1, left=1, color=col, height=0.6, alpha=0.9)
            ax.scatter(np.clip(v, 0.8, 1.3), i + rng.uniform(-0.18, 0.18, len(v)), s=9, color=TINTA2, zorder=3, lw=0)
            ax.text(1.43, i, f"{100 * (g - 1):+.1f} %".replace(".", ","), va="center", fontsize=8,
                    color=TINTA, ha="right", fontweight="bold" if r == "P" else "normal")
        ax.axvline(1, color=TINTA2, lw=1)
        ax.axvspan(1.31, 1.44, color="white", zorder=2)             # columna de etiquetas
        ax.set_xlim(0.8, 1.44)
        ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        ax.set_yticks(range(len(reps)), [SINT_TXT[r] for r in reps], fontsize=8)
        ax.set_title(m, loc="left", fontsize=9)
        ax.set_xlabel("MAE / MAE con X original")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ================================================================ 4. Búsqueda: controles, rango, algoritmos
def cargar_busqueda():
    """busqueda.csv + las filas del método completo (final.csv, P) con el mismo formato (variante 'CMA-ES')."""
    b = pd.read_csv(os.path.join(RES, "busqueda.csv"))
    f = pd.read_csv(os.path.join(RES, "final.csv"))
    p = f[f.representacion == "P"].pivot_table(index=["dataset", "semilla"], columns="modelo", values="mae")
    p.columns = ["mae " + x for x in p.columns]
    extra = f[f.representacion == "P"].drop_duplicates(["dataset", "semilla"]).set_index(["dataset", "semilla"])
    for col in ["fitness", "segundos", "lambdas", "pesos", "fitness0"]:
        p[col] = extra[col]
    p = p.reset_index().assign(variante="CMA-ES", v=1.0)
    orig = f[f.representacion == "original"].pivot_table(index=["dataset", "semilla"], columns="modelo", values="mae")
    orig.columns = ["orig " + x for x in orig.columns]
    return pd.concat([b, p], ignore_index=True).merge(orig.reset_index(), on=["dataset", "semilla"], how="left")


def contra_referencia(b, variantes, ref="CMA-ES", modelos=("KNN", "Regresión lineal", "Random Forest")):
    """Cociente MAE(variante)/MAE(ref) por dataset en las particiones comunes; Holm por modelo sobre las variantes."""
    filas = []
    r = b[b.variante == ref].set_index(["dataset", "semilla"])
    for v in variantes:
        a = b[b.variante == v].set_index(["dataset", "semilla"])
        comunes = a.index.intersection(r.index)
        if not len(comunes):
            continue
        fila = dict(variante=v, casos=len(comunes))
        for m in modelos:
            rel = (a.loc[comunes, "mae " + m] / r.loc[comunes, "mae " + m]).rename("rel").reset_index()
            s = resumen(rel)
            fila[m], fila["p " + m], fila["ds peor " + m] = s["cociente"], s["p"], s["ds"] - s["ds_mejora"]
        filas.append(fila)
    t = pd.DataFrame(filas).set_index("variante")
    for m in modelos:
        t["p_holm " + m] = holm(t["p " + m].values)
    return t


CONTROLES = ["solo pesos", "solo exponentes", "búsqueda aleatoria", "pesos MI", "YJ máx. verosimilitud + pesos MI"]
RANGOS = {0: "solo pesos", 1: "R=1", 2: "CMA-ES", 4: "R=4", 8: "R=8"}
ALGORITMOS = ["CMA-ES", "AG (propio)", "DE", "PSO", "BBO", "HS", "ABC", "búsqueda aleatoria"]
INICIO = ["inicio aleatorio", "sigma0=0.1", "sigma0=0.4", "población x2"]
# Tabla única de ablación de la búsqueda: todo frente a CMA-ES completo (θ0 = estandarización, σ0 = 0,2,
# población por defecto, λ ∈ [-2, 2]) en las mismas particiones
ABLACION = [("Piezas", "solo pesos", "solo pesos ($\\lambda=1$)"),
            ("Piezas", "solo exponentes", "solo exponentes ($w=1$)"),
            ("Sin búsqueda", "pesos MI", "pesos por inf.\\ mutua"),
            ("Sin búsqueda", "YJ máx. verosimilitud + pesos MI", "YJ (MV) + pesos inf.\\ mutua"),
            ("Búsqueda", "búsqueda aleatoria", "búsqueda aleatoria (1000 puntos)"),
            ("Inicio", "inicio aleatorio", "media inicial aleatoria"),
            ("Inicio", "sigma0=0.1", "$\\sigma_0=0{,}1$"),
            ("Inicio", "sigma0=0.4", "$\\sigma_0=0{,}4$"),
            ("Inicio", "población x2", "población $\\times2$"),
            ("Rango", "R=1", "$\\lambda\\in[-1,1]$"),
            ("Rango", "R=4", "$\\lambda\\in[-4,4]$"),
            ("Rango", "R=8", "$\\lambda\\in[-8,8]$")]


def tabla_rango(b):
    """λ ∈ [-R, R]: MAE de KNN frente a R = 2 (por dataset, Holm), fracción de λ aprendidos en el borde."""
    t = contra_referencia(b, [RANGOS[R] for R in [0, 1, 4, 8]], modelos=("KNN",))
    t.index = [f"R = {R}" for R in [0, 1, 4, 8]]
    borde = {}
    for R, v in RANGOS.items():
        if R == 0:
            continue
        s = b[(b.variante == v) & b.semilla.isin([40, 41, 42])]
        lam = np.concatenate([np.array(x.split("|"), float) for x in s.lambdas])
        borde[R] = float(np.mean(np.abs(lam) > 0.98 * R))
    return t, borde


def fig_rango(b, figsize=(3.6, 2.45)):
    ref = b[b.variante == "CMA-ES"].set_index(["dataset", "semilla"])["mae KNN"]
    xs, et = [0, 1, 2, 3, 4], ["0\n(sin YJ)", "1", "2", "4", "8"]
    curvas = {}
    for R in [0, 1, 2, 4, 8]:
        a = b[b.variante == RANGOS[R]].set_index(["dataset", "semilla"])["mae KNN"]
        if R == 2:
            a = a[a.index.get_level_values(1).isin([40, 41, 42])]
        com = a.index.intersection(ref.index)
        curvas[R] = (a.loc[com] / ref.loc[com]).rename("rel").reset_index().groupby("dataset").rel.apply(geo)
    por_ds = pd.DataFrame(curvas)
    fig, ax = plt.subplots(figsize=figsize)
    for _, fila in por_ds.iterrows():
        ax.plot(xs, fila.values, color=GRIS, lw=0.8, alpha=0.6)
    ax.plot(xs, [geo(por_ds[R].dropna()) for R in [0, 1, 2, 4, 8]], color=AZUL, lw=2, marker="o", ms=5,
            label="media geométrica")
    ax.plot([], [], color=GRIS, lw=0.8, label="cada dataset")
    ax.axhline(1, color=TINTA2, lw=0.8, ls="--")
    ax.set_xticks(xs, et)
    ax.set_xlabel("R  (λ ∈ [−R, R])")
    ax.set_ylabel("MAE KNN / MAE con R = 2")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def tabla_algoritmos(b):
    """Rango medio de fitness por partición (1 = mejor), cambio del MAE de KNN y R. lineal frente a X original.
    Solo particiones comunes a todos los algoritmos (semillas 40–41)."""
    algs = [a for a in ALGORITMOS if a in b.variante.unique()]
    s = b[b.variante.isin(algs) & b.semilla.isin([40, 41])].copy()
    n_alg = s.groupby(["dataset", "semilla"]).variante.nunique()
    s = s[s.set_index(["dataset", "semilla"]).index.isin(n_alg[n_alg == len(algs)].index)]
    s["rango"] = s.groupby(["dataset", "semilla"]).fitness.rank()
    t = s.groupby("variante").agg(rango=("rango", "mean"), fitness=("fitness", "mean"))
    for m in ["KNN", "Regresión lineal"]:
        s["rel"] = s["mae " + m] / s["orig " + m]
        t["cambio " + m] = 100 * (s.groupby("variante").apply(
            lambda g: geo(g.groupby("dataset").rel.apply(geo))) - 1)
    return t.loc[algs].sort_values("rango"), s[["dataset", "semilla"]].drop_duplicates().shape[0]


# ================================================================ 5. Coste
def tabla_coste(c):
    """Tiempo de ajuste de cada transformación (s por partición, mediana y máximo). Para P es el de la búsqueda."""
    f = cargar_reduccion()
    f = f[f.representacion == "P"].drop_duplicates(["dataset", "semilla"])
    filas = [dict(metodo="P (CMA-ES, 1000 evaluaciones)", mediana=f.segundos.median(), maximo=f.segundos.max())]
    for r in ["MLKR", "PG (gplearn)", "PCA v-CV", "YJ máx. verosimilitud", "cuantiles", "estándar + PCA"]:
        s = c[c.representacion == r].drop_duplicates(["dataset", "semilla"])
        if len(s) and "segundos_transf" in s:
            filas.append(dict(metodo=r, mediana=s.segundos_transf.median(), maximo=s.segundos_transf.max()))
    return pd.DataFrame(filas).set_index("metodo")


# ================================================================ 6. Diabetes
def fig_diabetes(d):
    f = d[d.dataset == "diabetes"]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    rng = np.random.default_rng(0)
    for i, m in enumerate(MODELOS):
        v = 100 * (f[(f.modelo == m) & (f.representacion == "P")].rel.values - 1)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=AZUL, edgecolor="white", lw=0.6, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=TINTA, lw=2, zorder=4)
    ax.axhline(0, color=TINTA2, lw=0.8, ls="--")
    ax.set_xticks(range(4), [CORTO_MOD[m] for m in MODELOS])
    ax.set_ylabel("cambio del MAE vs original (%)")
    ax.set_title("Diabetes: 5 particiones", loc="left", fontsize=9)
    fig.tight_layout()
    return fig


# ================================================================ 7. Exportar a LaTeX
def _num(x, dec=2):
    return f"{x:.{dec}f}".replace(".", "{,}")


def _pct(x, dec=1):
    return f"{x:+.{dec}f}".replace(".", "{,}").replace("-", "$-$").replace("+", "$+$") + "\\,\\%"


def _p(p):
    if not np.isfinite(p):
        return "--"
    return "$<0{,}001$" if p < 0.001 else "$" + f"{p:.3f}".replace(".", "{,}") + "$"


def exportar(d, c, b):
    m, tablas = {}, {}
    m["nDatasets"], m["nParticiones"] = str(d.dataset.nunique()), str(d.semilla.nunique())
    m["nCasos"] = str(len(d[(d.modelo == "KNN") & (d.representacion == "P")]))
    clave = {"Regresión lineal": "RL", "KNN": "KNN", "Árbol": "Arbol", "Random Forest": "RF"}

    # --- principal
    tp = tabla_principal(d)
    filas = []
    for mod, k in clave.items():
        r = tp.loc[mod]
        m[f"rel{k}"], m[f"cambio{k}"] = _num(r.cociente_orig), _pct(r.cambio_orig)
        m[f"casos{k}"], m[f"ds{k}"] = f"{int(r.casos_mejora_orig)}/{int(r.casos_orig)}", f"{int(r.ds_mejora_orig)}/{int(r.ds_orig)}"
        m[f"p{k}"] = _p(r.p_holm_orig)
        m[f"cambioVsEst{k}"], m[f"dsVsEst{k}"], m[f"pVsEst{k}"] = _pct(r.cambio_est), \
            f"{int(r.ds_mejora_est)}/{int(r.ds_est)}", _p(r.p_holm_est)
        m[f"peorDs{k}"] = _pct(100 * (r.peor_ds_orig - 1))
        filas.append(f"{CORTO_MOD[mod]} & {_num(r.estandar)} & {_num(r.sin_reducir)} & \\textbf{{{_num(r.cociente_orig)}}}"
                     f" & {int(r.ds_mejora_orig)}/{int(r.ds_orig)} & {_p(r.p_holm_orig)} & {_pct(r.cambio_est)}\\\\\n")
    tablas["filasPrincipal"] = filas
    for nombre, ef in [("Sint", efecto_sintesis(d)), ("Red", efecto_reduccion(d))]:
        for mod, k in clave.items():
            m[f"cambio{nombre}{k}"], m[f"p{nombre}{k}"] = _pct(ef.loc[mod, "cambio"]), _p(ef.loc[mod, "p_holm"])
            m[f"ds{nombre}{k}"] = f"{int(ef.loc[mod, 'ds_mejora'])}/{int(ef.loc[mod, 'ds'])}"
            m[f"rel{nombre}{k}"] = _num(ef.loc[mod, "cociente"])
    # --- reducción de dimensionalidad
    u = d[d.representacion == "P"].drop_duplicates(["dataset", "semilla"])
    m["dimMedia"] = _num(100 * (u.dim / u.n_atributos).mean(), 0) + "\\,\\%"
    m["fracEliminados"] = _num(100 * (1 - u.s / u.n_atributos).mean(), 0) + "\\,\\%"
    m["dimMin"], m["dimMax"] = _num(100 * (u.dim / u.n_atributos).min(), 0) + "\\,\\%", \
        _num(100 * (u.dim / u.n_atributos).max(), 0) + "\\,\\%"
    m["mMedia"] = _num(u.m.mean(), 1)
    m["fracConSintesis"] = _num(100 * (u.m > 0).mean(), 0) + "\\,\\%"
    m["tauMediana"] = _num(u.tau.median(), 2)
    m["casosNmenosUno"] = f"{int((u.s == u.n_atributos - 1).sum())}/{len(u)}"
    td = tabla_dimension(d)
    tablas["filasDimension"] = [f"{corto(i).replace('_', chr(92) + '_')} & {int(r.n)} & {_num(r.s, 1)} & "
                                f"{_num(r.m, 1)} & {_num(r.dim, 1)} & {_num(r.tau, 2)}\\\\\n" for i, r in td.iterrows()]
    W = np.array([list(map(float, x.split("|"))) for x in u[u.dataset == "california"].pesos]).mean(0)
    m["wLat"], m["wLon"], m["wPop"], m["wInc"] = _num(W[6]), _num(W[7]), _num(W[4]), _num(W[0])
    todos_w = np.concatenate([np.array(x.split("|"), float) for x in u.pesos])
    m["fracApagados"] = _num(100 * (todos_w < 0.1).mean(), 0) + "\\,\\%"

    # --- comparativa
    tc, ncomp = tabla_comparativa(c)
    m["nCompCasos"] = str(ncomp)
    nombres = {"estándar": "estándar", "YJ máx. verosimilitud": "YJ máx.\\ verosimilitud", "cuantiles": "cuantiles",
               "PCA 95 %": "solo PCA (95\\,\\%)", "PCA v-CV": "solo PCA (\\% por VC)",
               "estándar + PCA": "estándar + su PCA", "MLKR": "MLKR", "PG (gplearn)": "PG (gplearn)",
               "P sin reducir": "$P$ sin reducir", "P": "\\textbf{$P$}"}
    filas = []
    for r in tc.index:
        celdas = []
        for mod in MODELOS:
            v, pv = tc.loc[r, mod], tc.loc[r, "p " + mod]
            txt = _num(v)
            txt = "\\textbf{" + txt + "}" if np.isclose(v, tc[mod].min()) else txt
            celdas.append(txt + ("$^*$" if np.isfinite(pv) and pv < 0.05 else ""))
        filas.append(f"{nombres[r]} & " + " & ".join(celdas) + f" & {_num(tc.loc[r, 'rango medio'], 1)}"
                     f" & {_num(tc.loc[r, 'dim/n'], 2)}\\\\\n")
    tablas["filasComparativa"] = filas
    for r, k in [("MLKR", "MLKR"), ("PG (gplearn)", "PG"), ("YJ máx. verosimilitud", "YJMV"), ("cuantiles", "Cuant"),
                 ("estándar + PCA", "EstPCA"), ("PCA 95 %", "PCAnoventa"), ("PCA v-CV", "PCAcv"),
                 ("P sin reducir", "Psr"), ("P", "P")]:
        if r in tc.index:
            for mod, km in clave.items():
                m[f"comp{k}{km}"] = _pct(100 * (tc.loc[r, mod] - 1))
            m[f"rango{k}"] = _num(tc.loc[r, "rango medio"], 1)
    alt = tc.drop(index=[x for x in tc.index if x.startswith("P")])
    m["mejorAlt"] = alt["rango medio"].idxmin()
    m["rangoMejorAlt"] = _num(alt["rango medio"].min(), 1)
    m["nReps"] = str(len(tc))

    # --- síntesis: variante con v
    if len(b) and "CMA-ES + v" in set(b.variante):
        bv = b[b.variante == "CMA-ES + v"]
        m["vMedia"] = _num(bv.v.mean(), 2)
        m["vFracUno"] = _num(100 * (bv.v > 0.98).mean(), 0) + "\\,\\%"
        m["nSintMedia"] = _num(bv.n_sintesis.mean(), 1)

    # --- ablación de la búsqueda (piezas, inicio, rango) en una sola tabla
    filas = []
    disponibles = [v for _, v, _ in ABLACION if len(b) and v in set(b.variante)]
    if disponibles:
        t = contra_referencia(b, disponibles)
        grupo_prev = None
        for grupo, v, txt in ABLACION:
            if v not in t.index:
                continue
            if grupo != grupo_prev:
                filas.append(("\\midrule\n" if grupo_prev else "") + f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{grupo}}}}}\\\\\n")
                grupo_prev = grupo
            celdas = [f"{_pct(100 * (t.loc[v, mm] - 1))}{'$^*$' if t.loc[v, 'p_holm ' + mm] < 0.05 else ''}"
                      for mm in ["KNN", "Regresión lineal", "Random Forest"]]
            filas.append(f"\\quad {txt} & " + " & ".join(celdas) + "\\\\\n")
        for v, k in [("inicio aleatorio", "IniAleat"), ("sigma0=0.1", "SigmaBajo"), ("sigma0=0.4", "SigmaAlto"),
                     ("población x2", "PobDoble")]:
            if v in t.index:
                m[f"ctrl{k}"], m[f"pCtrl{k}"] = _pct(100 * (t.loc[v, "KNN"] - 1)), _p(t.loc[v, "p_holm KNN"])
                m[f"ctrl{k}RL"] = _pct(100 * (t.loc[v, "Regresión lineal"] - 1))
        m["nAblCasos"] = str(int(t.casos.max()))
    tablas["filasAblacion"] = filas

    # --- controles
    filas = []
    if len(b) and set(CONTROLES) & set(b.variante):
        t = contra_referencia(b, [v for v in CONTROLES if v in set(b.variante)])
        txt = {"solo pesos": "solo pesos ($\\lambda=1$)", "solo exponentes": "solo exponentes ($w=1$)",
               "búsqueda aleatoria": "búsqueda aleatoria", "pesos MI": "pesos por inf.\\ mutua",
               "YJ máx. verosimilitud + pesos MI": "YJ (MV) + pesos inf.\\ mutua"}
        cl = {"solo pesos": "Pesos", "solo exponentes": "Exp", "búsqueda aleatoria": "Aleat", "pesos MI": "MI",
              "YJ máx. verosimilitud + pesos MI": "YJMI"}
        for v in t.index:
            celdas = [f"{_pct(100 * (t.loc[v, mm] - 1))}{'$^*$' if t.loc[v, 'p_holm ' + mm] < 0.05 else ''}"
                      for mm in ["KNN", "Regresión lineal", "Random Forest"]]
            filas.append(f"{txt[v]} & " + " & ".join(celdas) + "\\\\\n")
            m[f"ctrl{cl[v]}"], m[f"ctrl{cl[v]}RL"] = _pct(100 * (t.loc[v, "KNN"] - 1)), \
                _pct(100 * (t.loc[v, "Regresión lineal"] - 1))
            m[f"pCtrl{cl[v]}"] = _p(t.loc[v, "p_holm KNN"])
        m["nCtrlCasos"] = str(int(t.casos.max()))
    tablas["filasControles"] = filas

    # --- rango
    if len(b) and "R=8" in set(b.variante):
        tr, borde = tabla_rango(b)
        for R, k in [(0, "Cero"), (1, "Uno"), (4, "Cuatro"), (8, "Ocho")]:
            m[f"rango{k}R"] = _pct(100 * (tr.loc[f"R = {R}", "KNN"] - 1))
            m[f"pRango{k}"] = _p(tr.loc[f"R = {R}", "p_holm KNN"])
            m[f"dsPeorRango{k}"] = str(int(tr.loc[f"R = {R}", "ds peor KNN"]))
        for R, k in [(1, "Uno"), (2, "Dos"), (4, "Cuatro"), (8, "Ocho")]:
            m[f"borde{k}"] = _num(100 * borde[R], 0) + "\\,\\%"
        m["nRango"] = str(int(tr.casos.iloc[0]))
    mle = pd.read_csv(os.path.join(RES, "lambdas_mle.csv"))
    m["mleFueraDos"] = _num(100 * (mle.lam_mle.abs() > 2).mean(), 0) + "\\,\\%"

    # --- algoritmos
    filas = []
    if len(b) and "PSO" in set(b.variante):
        ta, nalg = tabla_algoritmos(b)
        m["nAlgCasos"] = str(nalg)
        for a in ta.index:
            nom = "\\textbf{CMA-ES}" if a == "CMA-ES" else a
            filas.append(f"{nom} & {_num(ta.loc[a, 'rango'], 1)} & {_pct(ta.loc[a, 'cambio KNN'])} & "
                         f"{_pct(ta.loc[a, 'cambio Regresión lineal'])}\\\\\n")
        m["rangoCMA"], m["algMejor"] = _num(ta.loc["CMA-ES", "rango"], 1), ta.index[0]
        m["algKNNcma"] = _pct(ta.loc["CMA-ES", "cambio KNN"])
        m["algKNNaleat"] = _pct(ta.loc["búsqueda aleatoria", "cambio KNN"])
        otros = ta.drop(index=["CMA-ES", "búsqueda aleatoria"])["cambio KNN"]
        m["algKNNmin"], m["algKNNmax"] = _pct(otros.min()), _pct(otros.max())
    tablas["filasAlgoritmos"] = filas

    # --- coste
    tco = tabla_coste(c)
    fila_p = tco.loc["P (CMA-ES, 1000 evaluaciones)"]
    m["segMediana"], m["segMax"] = _num(fila_p.mediana, 0), _num(fila_p.maximo, 0)
    m["msPorEval"] = _num(fila_p.mediana, 0)        # s por 1000 evaluaciones = ms por evaluación
    for r, k in [("MLKR", "MLKR"), ("PG (gplearn)", "PG"), ("YJ máx. verosimilitud", "YJMV")]:
        if r in tco.index:
            m[f"seg{k}"] = _num(tco.loc[r, "mediana"], 1)
    tablas["filasCoste"] = [f"{i.replace('%', chr(92) + '%')} & {_num(r.mediana, 1)} & {_num(r.maximo, 1)}\\\\\n"
                            for i, r in tco.iterrows()]

    # --- comportamiento de CMA-ES (generaciones, selección, parada)
    ruta = os.path.join(RES, "parada.csv")
    if os.path.exists(ruta):
        pa = pd.read_csv(ruta)
        m["popMin"], m["popMax"] = str(int(pa.lambda_pob.min())), str(int(pa.lambda_pob.max()))
        m["muMin"], m["muMax"] = str(int(pa.mu.min())), str(int(pa.mu.max()))
        m["muEffMin"], m["muEffMax"] = _num(pa.mu_eff.min(), 1), _num(pa.mu_eff.max(), 1)
        m["genMin"], m["genMax"] = str(int(pa.generaciones.min())), str(int(pa.generaciones.max()))
        m["nParadaPresupuesto"] = str(int(pa.parada.str.contains("maxfevals").sum()))
        m["nParadaTotal"] = str(len(pa))
        m["evalMejorMediana"] = str(int(pa.eval_mejor.median()))
        m["evalMejorMin"] = str(int(pa.eval_mejor.min()))
        m["mejoraFitnessMediana"] = _pct(100 * (pa.fitness_final / pa.fitness_inicial - 1).median())
        m["sigmaFinMin"], m["sigmaFinMax"] = _num(pa.sigma_final.min(), 3), _num(pa.sigma_final.max(), 2)

    # --- contraste: reducción con regla fija DESPUÉS de la evolución (reduccion/cribado_posthoc.py)
    ruta = os.path.join(os.path.dirname(RED), "cribado_posthoc.csv")
    if os.path.exists(ruta):
        cr = pd.read_csv(ruta)
        p = cr.pivot_table(index=["dataset", "semilla", "modelo"], columns="representacion", values="mae")
        rel = np.log(p.div(p["original"], axis=0))
        for rep_, k in [("sel τ*", "Sel"), ("sel w<0.1", "Fijo"), ("pca q*", "Pca")]:
            for mod, km in clave.items():
                m[f"crib{k}{km}"] = _pct(100 * (np.exp(rel.xs(mod, level="modelo")[rep_].mean()) - 1))
            u_ = cr[cr.representacion == rep_].drop_duplicates(["dataset", "semilla"])
            m[f"cribDim{k}"] = _num(100 * (u_.dim / u_.n).mean(), 0) + "\\,\\%"
        u_ = cr[cr.representacion == "sel w<0.1"].drop_duplicates(["dataset", "semilla"])
        m["cribFijoNoReduce"] = f"{int((u_.dim == u_.n).sum())}/{len(u_)}"

    # --- datasets
    ds = d.drop_duplicates("dataset").set_index("dataset")[["n_filas", "n_atributos"]]
    tablas["filasDatasets"] = [f"{corto(n).replace('_', chr(92) + '_')} & {r.n_filas:,} & {r.n_atributos}\\\\\n"
                               .replace(",", "\\,") for n, r in ds.iterrows()]

    with open(os.path.join(CARPETA, "numeros.tex"), "w") as fh:
        fh.write("% Generado por analisis.py: NO editar a mano.\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(CARPETA, "tablas.tex"), "w") as fh:
        fh.write("% Generado por analisis.py: filas de las tablas (una macro por tabla).\n")
        for k, filas in tablas.items():
            fh.write(f"\\newcommand{{\\{k}}}{{%\n" + ("".join(filas) or "pendiente\\\\\n") + "}\n")
    return m


def todas():
    d = cargar_final()
    c = cargar_comparativa()
    hay_b = os.path.exists(os.path.join(RES, "busqueda.csv"))
    b = cargar_busqueda() if hay_b else pd.DataFrame(columns=["variante"])
    guardar(fig_principal(d), "principal")
    f1, f2 = figs_poster(d)
    guardar(f1, "poster_barras")
    guardar(f2, "poster_heatmap")
    guardar(fig_comparativa_cd(c), "comparativa_cd")
    guardar(fig_comparativa_cd(c, modelos=["KNN", "Random Forest"], figsize=(10.5, 3.0)), "comparativa_cd2")
    guardar(fig_comparativa_cd(c, modelos=["KNN", "Random Forest"], figsize=(6.0, 5.2), columnas=1), "poster_cd")
    guardar(fig_sintesis(c), "sintesis")
    guardar(fig_sintesis(c, modelos=("Random Forest",), figsize=(5.2, 3.0)), "sintesis_rf")
    guardar(fig_diabetes(d), "diabetes")
    if hay_b and "R=8" in set(b.variante):
        guardar(fig_rango(b), "rango_yj_col")
    plt.close("all")
    return exportar(d, c, b)


if __name__ == "__main__":
    matplotlib.use("Agg")
    todas()
    print("figuras en", FIG, "; cifras en numeros.tex y tablas.tex")
