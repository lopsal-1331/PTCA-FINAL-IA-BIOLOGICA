"""
analisis.py — Tablas, contrastes, figuras y cifras de la memoria, el paper y el póster a partir de resultados/*.csv.

    python analisis.py      -> figuras/*.pdf|png, numeros.tex (macros con las cifras) y tablas.tex (filas de tablas)

Convención: cociente e'/e = MAE con X' / MAE con X (< 1 mejora), media geométrica sobre las particiones de cada
dataset y luego sobre los 12 datasets. Contrastes por dataset (n = 12): Wilcoxon con corrección de Holm; comparaciones
múltiples: Friedman y diferencia crítica de Nemenyi (Demšar, 2006). Ninguna cifra de los documentos se escribe a mano.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import ptca

RES, FIG = os.path.join(ptca.CARPETA, "resultados"), os.path.join(ptca.CARPETA, "figuras")
MODELOS = ["Regresión lineal", "KNN", "Árbol", "Random Forest"]
CORTO = {"Regresión lineal": "R. lineal", "KNN": "KNN", "Árbol": "Árbol", "Random Forest": "RF"}
CLAVE = {"Regresión lineal": "RL", "KNN": "KNN", "Árbol": "Arbol", "Random Forest": "RF"}
AZUL, NARANJA, GRIS, TINTA, TINTA2 = "#2E6DB4", "#D9731A", "#A3A9AF", "#1D2329", "#5A6772"
DIVERGENTE = LinearSegmentedColormap.from_list("div", [AZUL, "#F2F2F0", NARANJA])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#ECEEF0", "axes.axisbelow": True,
                     "axes.edgecolor": "#C9CED3", "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "savefig.bbox": "tight", "legend.frameon": False})


def nombre(d):
    return {"concrete_compressive_strength": "concrete"}.get(d, d)


def guardar(fig, n):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, n + ".pdf"))
    fig.savefig(os.path.join(FIG, n + ".png"), dpi=200)
    plt.close(fig)


# ================================================================================================================ datos
def leer(n):
    return pd.read_csv(os.path.join(RES, n + ".csv"))


def cocientes(d, ref="original"):
    """Tabla (dataset, semilla, modelo) × representación con MAE / MAE de la referencia."""
    p = d.pivot_table(index=["dataset", "semilla", "modelo"], columns="representacion", values="mae")
    return p.div(p[ref], axis=0)


def resumen_modelo(rel, rep, modelos=MODELOS):
    """Para cada modelo: contraste por dataset de la columna rep de una tabla de cocientes."""
    t = pd.DataFrame({m: ptca.contraste(rel.xs(m, level="modelo")[rep].dropna()) for m in modelos}).T
    t["p_holm"] = ptca.holm(t.p.astype(float).values)
    return t


# ============================================================================================================ tablas
def tabla_principal():
    """P frente a X original y frente a X estandarizado (Holm sobre los 8 contrastes)."""
    d = leer("confirmacion")
    r_o, r_e = cocientes(d, "original"), cocientes(d, "estándar")
    a, b = resumen_modelo(r_o, "P"), resumen_modelo(r_e, "P")
    t = pd.DataFrame({"estándar": [ptca.geo(ptca.por_dataset(r_o.xs(m, level="modelo")["estándar"])) for m in MODELOS]},
                     index=MODELOS)
    for c in ["cociente", "cambio", "ds_mejora", "ds", "p", "peor_ds", "peor"]:
        t[c] = a[c]
    t["vs_est"], t["p_est"] = b.cambio, b.p
    aj = ptca.holm(np.r_[t.p.astype(float), t.p_est.astype(float)])
    t["p_holm"], t["p_est_holm"] = aj[:4], aj[4:]
    return t


def tabla_sensibilidad(excluir=("kin8nm",)):
    d = leer("confirmacion")
    rel = cocientes(d[~d.dataset.isin(excluir)])
    return resumen_modelo(rel, "P")


def tabla_dimension():
    d = leer("confirmacion")
    u = d[d.representacion == "P"].drop_duplicates(["dataset", "semilla"])
    t = u.groupby("dataset").agg(n=("n", "first"), orig=("n_orig", "mean"), comp=("n_comp", "mean"), dim=("dim", "mean"))
    return t.loc[ptca.DATASETS], u


ORDEN_ABL = ["fitness solo KNN", "sin síntesis", "sin forma (λ = 1)", "sin límite n−1", "búsqueda aleatoria",
             "algoritmo genético"]


def tabla_ablacion():
    """Cada variante frente a P en las mismas particiones (semillas 40–41): cociente por modelo (> 1 = peor que P),
    Holm por modelo sobre las variantes; dimensión media y segundos."""
    a, c = leer("ablacion"), leer("confirmacion")
    d = pd.concat([a, c[c.semilla.isin(a.semilla.unique()) & (c.representacion == "P")]])
    rel = cocientes(d, "P")
    vs = [v for v in ORDEN_ABL if v in rel.columns]
    filas = {}
    for v in vs:
        filas[v] = {m: ptca.contraste(rel.xs(m, level="modelo")[v].dropna()) for m in MODELOS}
    t = pd.DataFrame(index=vs)
    for m in MODELOS:
        t[m] = [filas[v][m]["cociente"] for v in vs]
        t["p " + m] = ptca.holm([filas[v][m]["p"] for v in vs])
    t["media"] = [ptca.geo([filas[v][m]["cociente"] for m in MODELOS]) for v in vs]
    u = a.drop_duplicates(["dataset", "semilla", "representacion"])
    t["dim/n"] = (u.dim / u.n).groupby(u.representacion).mean()
    t["segundos"] = u.groupby("representacion").segundos.median()
    return t, d[["dataset", "semilla"]].drop_duplicates().shape[0]


ORDEN_CRIB = ["P", "fitness solo KNN", "fitness minimax", "familia", "parada (paciencia 15)", "parada (paciencia 30)",
              "elegir θ con apartado"]


def tabla_cribado():
    """Exploración (semillas 0–1, 7 datasets): e'/e por modelo, media de los 4, peor modelo y tiempo. 'familia' =
    cada modelo con la X' aprendida con su propio fitness (KNN, RL, árbol; RF con un RF de 5 árboles)."""
    d = leer("cribado")
    rel = cocientes(d)
    por = np.log(rel).groupby(["dataset", "modelo"]).mean()
    g = np.exp(por.groupby("modelo").mean()).T
    fam = {"KNN": "fitness solo KNN", "Regresión lineal": "familia RL", "Árbol": "familia árbol",
           "Random Forest": "familia RF"}
    g.loc["familia"] = [g.loc[fam[m], m] for m in g.columns]
    g = g.loc[[r for r in ORDEN_CRIB if r in g.index], MODELOS]
    g["media"] = np.exp(np.log(g[MODELOS]).mean(axis=1))
    g["peor"] = g[MODELOS].max(axis=1)
    u = d[d.segundos.notna()].drop_duplicates(["dataset", "semilla", "representacion"])
    seg = u.groupby("representacion").segundos.mean()
    seg["familia"] = np.mean([seg[fam[m]] for m in MODELOS])
    g["segundos"] = seg
    g["evaluaciones"] = u.groupby("representacion").evaluaciones.mean()
    g.loc["familia", "evaluaciones"] = 1000
    return g, d[["dataset", "semilla"]].drop_duplicates().shape[0]


ORDEN_COMP = ["estándar", "YJ máx. verosimilitud", "cuantiles", "PCA 95 %", "PCA v-CV", "estándar + PCA", "MLKR",
              "PG (gplearn)", "P"]


def tabla_comparativa():
    c, f = leer("comparativa"), leer("confirmacion")
    d = pd.concat([c, f[f.representacion.isin(["original", "P"])]])
    d = d[d.representacion != "estándar"] if False else d
    rel = cocientes(d.drop_duplicates(["dataset", "semilla", "representacion", "modelo"]))
    reps = [r for r in ORDEN_COMP if r in rel.columns]
    t = pd.DataFrame(index=reps)
    for m in MODELOS:
        pd_ = rel.xs(m, level="modelo")[reps].groupby(level="dataset").apply(lambda x: np.exp(np.log(x).mean()))
        t[m] = np.exp(np.log(pd_).mean())
        rangos, _, _ = ptca.friedman_nemenyi(pd_)
        t["rango " + m] = rangos
        ps = [np.nan if r == "P" else ptca.contraste(np.log(pd_["P"] / pd_[r]).pipe(np.exp).rename_axis("dataset")
                                                    .to_frame("x").assign(semilla=0).set_index("semilla", append=True).x)["p"]
              for r in reps]
        ps = np.array(ps, float)
        aj = np.full_like(ps, np.nan)
        aj[~np.isnan(ps)] = ptca.holm(ps[~np.isnan(ps)])
        t["p " + m] = aj
    t["rango"] = t[[f"rango {m}" for m in MODELOS]].mean(axis=1)
    dims = d.drop_duplicates(["dataset", "semilla", "representacion"])
    dims = dims.assign(f=dims.dim / dims.n)
    t["dim/n"] = dims.groupby("representacion").f.mean()
    t.loc["P", "dim/n"] = dims[dims.representacion == "P"].f.mean()
    return t


# ============================================================================================================ figuras
def fig_principal():
    d = leer("confirmacion")
    rel = cocientes(d)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1]))
    x, w = np.arange(4), 0.36
    for i, (rep, col, et) in enumerate([("estándar", GRIS, "X estandarizado"), ("P", AZUL, "P (dim X' < n)")]):
        v = [ptca.geo(ptca.por_dataset(rel.xs(m, level="modelo")[rep])) for m in MODELOS]
        a1.bar(x + (i - 0.5) * w, np.array(v) - 1, w * 0.92, bottom=1, color=col, label=et)
        if rep == "P":
            for xi, vi in zip(x, v):
                a1.text(xi + w / 2, vi - 0.01 if vi < 1 else vi + 0.004, f"{100 * (vi - 1):+.1f} %".replace(".", ","),
                        ha="center", va="top" if vi < 1 else "bottom", fontsize=8.5)
    a1.axhline(1, color=TINTA2, lw=1)
    a1.set_xticks(x, ["Regresión\nlineal", "KNN", "Árbol", "Random\nForest"])
    a1.set_ylabel("MAE / MAE con X original  (< 1 mejora)")
    a1.set_ylim(0.7, 1.04)
    a1.legend(loc="lower left")
    a1.set_title("e'/e por modelo (media geométrica, 12 datasets)", loc="left", fontsize=9)
    t = rel["P"].groupby(["dataset", "modelo"]).apply(ptca.geo).unstack()[MODELOS]
    t = t.loc[t.mean(axis=1).sort_values().index]
    lim = max(abs(np.log(t.values)).max(), 0.05)
    a2.imshow(np.log(t.values), cmap=DIVERGENTE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    a2.set_xticks(range(4), [CORTO[m] for m in MODELOS])
    a2.set_yticks(range(len(t)), [nombre(i) for i in t.index])
    a2.grid(False)
    for i in range(t.shape[0]):
        for j in range(4):
            v = t.values[i, j]
            a2.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=7.5,
                    color="white" if abs(np.log(v)) > 0.6 * lim else TINTA)
    a2.set_title("e'/e de P por dataset (azul mejora, naranja empeora)", loc="left", fontsize=9)
    fig.tight_layout()
    return fig


def fig_barras_tabla(t, cols, etiquetas, destacar, xlabel, titulo=None, figsize=(6.2, 3.0), lim=(0.85, 1.1)):
    """Puntos por modelo + media para cada fila de una tabla de cocientes (cribado, ablación)."""
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(t))
    marcas = {"Regresión lineal": "s", "KNN": "o", "Árbol": "^", "Random Forest": "D"}
    for m in cols:
        ax.scatter(t[m], y, marker=marcas[m], s=26, color=TINTA2, label=CORTO[m], zorder=3, alpha=0.8)
    ax.barh(y, t["media"] - 1, left=1, color=[AZUL if r == destacar else GRIS for r in t.index], height=0.55,
            alpha=0.85, label="media de los 4")
    ax.axvline(1, color=TINTA2, lw=1)
    ax.set_yticks(y, [etiquetas.get(r, r) for r in t.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(*lim)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.22))
    ax.grid(axis="y", visible=False)
    if titulo:
        ax.set_title(titulo, loc="left", fontsize=9)
    fig.tight_layout()
    return fig


def fig_cd(t_modelos=("KNN", "Regresión lineal")):
    c, f = leer("comparativa"), leer("confirmacion")
    d = pd.concat([c, f[f.representacion.isin(["original", "P"])]])
    rel = cocientes(d)
    fig, ejes = plt.subplots(len(t_modelos), 1, figsize=(6.2, 2.6 * len(t_modelos)))
    for ax, m in zip(np.atleast_1d(ejes), t_modelos):
        pd_ = rel.xs(m, level="modelo")[ORDEN_COMP].groupby(level="dataset").apply(lambda x: np.exp(np.log(x).mean()))
        rangos, p, cd = ptca.friedman_nemenyi(pd_)
        k = len(rangos)
        ax.set_xlim(k + 0.3, 0.7)
        ax.set_ylim(0, 1.08)
        ax.axis("off")
        ax.hlines(0.86, 1, k, color=TINTA2)
        for i in range(1, k + 1):
            ax.vlines(i, 0.86, 0.9, color=TINTA2)
            ax.text(i, 0.93, str(i), ha="center", fontsize=7)
        ax.hlines(1.02, 1, 1 + cd, color=TINTA, lw=1.5)
        ax.text(1 + cd / 2, 1.04, f"CD = {cd:.2f}".replace(".", ","), ha="center", fontsize=7)
        mitad = (k + 1) // 2
        for i, (nm, ri) in enumerate(rangos.items()):
            der = i < mitad
            yy = 0.72 - 0.11 * (i if der else i - mitad)
            xt = 0.75 if der else k + 0.25
            col = AZUL if nm == "P" else TINTA
            ax.plot([ri, ri, xt], [0.86, yy, yy], color=col, lw=0.9)
            ax.text(xt + (-0.05 if der else 0.05), yy, f"{nm} ({ri:.1f})".replace(".", ","),
                    ha="left" if der else "right", va="center", fontsize=7, color=col)
        ax.set_title(f"{m} (Friedman p = {p:.1e})".replace(".", ","), loc="left", fontsize=9)
    fig.tight_layout()
    return fig


# ============================================================================================================== LaTeX
def _n(x, dec=2):
    return f"{x:.{dec}f}".replace(".", "{,}")


def _pc(x, dec=1):
    return f"{x:+.{dec}f}".replace(".", "{,}").replace("-", "$-$").replace("+", "$+$") + "\\,\\%"


def _p(p):
    return "--" if not np.isfinite(p) else ("$<0{,}001$" if p < 0.001 else "$" + f"{p:.3f}".replace(".", "{,}") + "$")


def _esc(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("λ", "$\\lambda$").replace("θ", "$\\theta$")\
        .replace("−", "$-$")


def exportar():
    m, tb = {}, {}
    tp = tabla_principal()
    m["nDatasets"], m["nParticiones"] = "12", "5"
    filas = []
    for mod, k in CLAVE.items():
        r = tp.loc[mod]
        m[f"cambio{k}"], m[f"ds{k}"], m[f"p{k}"] = _pc(r.cambio), f"{int(r.ds_mejora)}/{int(r.ds)}", _p(r.p_holm)
        m[f"pSinHolm{k}"] = _p(r.p)
        m[f"vsEst{k}"], m[f"pVsEst{k}"] = _pc(r.vs_est), _p(r.p_est_holm)
        m[f"peor{k}"], m[f"peorDs{k}"] = _pc(100 * (r.peor - 1)), _esc(nombre(r.peor_ds))
        filas.append(f"{CORTO[mod]} & {_n(r['estándar'])} & \\textbf{{{_n(r.cociente)}}} & {int(r.ds_mejora)}/{int(r.ds)}"
                     f" & {_p(r.p_holm)} & {_pc(r.vs_est)} & {_p(r.p_est_holm)}\\\\\n")
    tb["filasPrincipal"] = filas
    m["cambioMedio"] = _pc(100 * (ptca.geo(tp.cociente.astype(float)) - 1))
    ts = tabla_sensibilidad()
    for mod, k in CLAVE.items():
        m[f"sinKin{k}"], m[f"pSinKin{k}"] = _pc(ts.loc[mod, "cambio"]), _p(ts.loc[mod, "p_holm"])
    m["sinKinMedio"] = _pc(100 * (ptca.geo(ts.cociente.astype(float)) - 1))
    # máximas reducciones
    d = leer("confirmacion")
    rel = cocientes(d)
    g = rel["P"].groupby(["dataset", "modelo"]).apply(ptca.geo)
    for mod, k in CLAVE.items():
        gm = g.xs(mod, level="modelo")
        m[f"max{k}"], m[f"maxDs{k}"] = _pc(100 * (gm.min() - 1)), _esc(nombre(gm.idxmin()))
    ds_, mod_ = g.idxmin()
    sel = lambda rep: d[(d.dataset == ds_) & (d.modelo == mod_) & (d.representacion == rep)].mae.mean()
    m["maxMaeOrig"], m["maxMaeP"] = _n(sel("original"), 2), _n(sel("P"), 2)
    # dimensión
    td, u = tabla_dimension()
    m["dimMedia"] = _n(100 * (u.dim / u.n).mean(), 0) + "\\,\\%"
    m["origMedia"], m["compMedia"] = _n(u.n_orig.mean(), 1), _n(u.n_comp.mean(), 1)
    m["fracComp"] = _n(100 * (u.n_comp > 0).mean(), 0) + "\\,\\%"
    m["fracElim"] = _n(100 * (1 - u.n_orig / u.n).mean(), 0) + "\\,\\%"
    m["siempreMenor"] = "sí" if (u.dim < u.n).all() else "no"
    m["segMediana"], m["segMax"] = _n(u.segundos.median(), 0), _n(u.segundos.max(), 0)
    tb["filasDimension"] = [f"{_esc(nombre(i))} & {int(r.n)} & {_n(r.orig, 1)} & {_n(r.comp, 1)} & {_n(r.dim, 1)}\\\\\n"
                            for i, r in td.iterrows()]
    # cribado
    gc, ncr = tabla_cribado()
    m["cribCasos"] = str(ncr)
    nombres_c = {"P": "\\textbf{$P$: KNN+RL+árbol}", "fitness solo KNN": "fitness solo KNN",
                 "fitness minimax": "fitness minimax", "familia": "un fitness por familia",
                 "parada (paciencia 15)": "+ parada (pac.\\ 15)", "parada (paciencia 30)": "+ parada (pac.\\ 30)",
                 "elegir θ con apartado": "+ elegir $\\theta$ con apartado"}
    tb["filasCribado"] = [f"{nombres_c[r]} & " + " & ".join(_n(gc.loc[r, x]) for x in MODELOS) +
                          f" & {_n(gc.loc[r, 'media'], 3)} & {_n(gc.loc[r, 'peor'], 3)} & {_n(gc.loc[r, 'segundos'], 0)}\\\\\n"
                          for r in gc.index]
    for r, k in [("P", "P"), ("fitness solo KNN", "Knn"), ("fitness minimax", "Minimax"), ("familia", "Familia"),
                 ("parada (paciencia 15)", "Parada"), ("parada (paciencia 30)", "ParadaTreinta"),
                 ("elegir θ con apartado", "Seleccion")]:
        if r in gc.index:
            m[f"crib{k}"], m[f"crib{k}Seg"] = _n(gc.loc[r, "media"], 3), _n(gc.loc[r, "segundos"], 0)
            m[f"crib{k}KNN"] = _n(gc.loc[r, "KNN"], 3)
    # ablación
    if os.path.exists(os.path.join(RES, "ablacion.csv")):
        ta, nab = tabla_ablacion()
        m["ablCasos"] = str(nab)
        tb["filasAblacion"] = [f"{_esc(v)} & " + " & ".join(
            _pc(100 * (ta.loc[v, x] - 1)) + ("$^*$" if ta.loc[v, 'p ' + x] < 0.05 else "") for x in MODELOS) +
            f" & {_pc(100 * (ta.loc[v, 'media'] - 1))} & {_n(ta.loc[v, 'dim/n'])}\\\\\n" for v in ta.index]
        for v, k in [("fitness solo KNN", "Knn"), ("sin síntesis", "SinSint"), ("sin forma (λ = 1)", "SinForma"),
                     ("sin límite n−1", "SinLimite"), ("búsqueda aleatoria", "Aleat"), ("algoritmo genético", "AG")]:
            if v in ta.index:
                m[f"abl{k}"] = _pc(100 * (ta.loc[v, "media"] - 1))
                for x, kx in CLAVE.items():
                    m[f"abl{k}{kx}"] = _pc(100 * (ta.loc[v, x] - 1))
                m[f"abl{k}Dim"] = _n(ta.loc[v, "dim/n"])
    # comparativa
    tc = tabla_comparativa()
    nombres = {"estándar": "estándar", "YJ máx. verosimilitud": "YJ máx.\\ verosimilitud", "cuantiles": "cuantiles",
               "PCA 95 %": "solo PCA (95\\,\\%)", "PCA v-CV": "solo PCA (\\% por VC)", "estándar + PCA": "estándar + su PCA",
               "MLKR": "MLKR", "PG (gplearn)": "prog.\\ genética", "P": "\\textbf{$P$}"}
    filas = []
    for r in tc.index:
        celdas = []
        for x in MODELOS:
            v, pv = tc.loc[r, x], tc.loc[r, "p " + x]
            s = _n(v)
            s = "\\textbf{" + s + "}" if np.isclose(v, tc[x].min()) else s
            celdas.append(s + ("$^*$" if np.isfinite(pv) and pv < 0.05 else ""))
        filas.append(f"{nombres[r]} & " + " & ".join(celdas) + f" & {_n(tc.loc[r, 'rango'], 1)} & {_n(tc.loc[r, 'dim/n'])}\\\\\n")
    tb["filasComparativa"] = filas
    orden = tc.rango.sort_values()
    m["rangoP"], m["posP"] = _n(tc.loc["P", "rango"], 1), str(list(orden.index).index("P") + 1)
    for r, k in [("PG (gplearn)", "PG"), ("estándar + PCA", "EstPCA"), ("MLKR", "MLKR"), ("PCA 95 %", "PCA"),
                 ("YJ máx. verosimilitud", "YJMV")]:
        m[f"rango{k}"] = _n(tc.loc[r, "rango"], 1)
        for x, kx in CLAVE.items():
            m[f"comp{k}{kx}"] = _pc(100 * (tc.loc[r, x] - 1))
    with open(os.path.join(ptca.CARPETA, "numeros.tex"), "w") as fh:
        fh.write("% Generado por analisis.py: NO editar a mano.\n")
        fh.writelines(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())
    with open(os.path.join(ptca.CARPETA, "tablas.tex"), "w") as fh:
        fh.write("% Generado por analisis.py.\n")
        fh.writelines(f"\\newcommand{{\\{k}}}{{%\n" + "".join(v) + "}\n" for k, v in tb.items())
    return m


ET_CRIB = {"P": "P: fitness KNN+RL+árbol", "fitness solo KNN": "fitness solo KNN", "fitness minimax": "fitness minimax",
           "familia": "un fitness por familia", "parada (paciencia 15)": "+ parada (paciencia 15)",
           "parada (paciencia 30)": "+ parada (paciencia 30)", "elegir θ con apartado": "+ elegir θ con apartado"}


def todas():
    guardar(fig_principal(), "principal")
    gc, _ = tabla_cribado()
    guardar(fig_barras_tabla(gc, MODELOS, ET_CRIB, "P", "MAE / MAE con X original (exploración)"), "cribado")
    if os.path.exists(os.path.join(RES, "ablacion.csv")):
        ta, _ = tabla_ablacion()
        ta = pd.concat([pd.DataFrame({**{x: [1.0] for x in MODELOS}, "media": [1.0]}, index=["P"]), ta])
        guardar(fig_barras_tabla(ta, MODELOS, {"P": "P (método)"}, "P", "MAE / MAE de P  (> 1: peor que P)",
                                 lim=(0.85, 1.35)), "ablacion")
    guardar(fig_cd(), "comparativa_cd")
    return exportar()


if __name__ == "__main__":
    matplotlib.use("Agg")
    todas()
    print("figuras en", FIG, "; cifras en numeros.tex y tablas.tex")
