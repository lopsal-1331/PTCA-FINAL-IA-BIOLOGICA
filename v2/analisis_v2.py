"""
analisis_v2.py — Cifras, tablas y figuras de P v2. Reutiliza la estadística de final/analisis.py.

Fuentes (todas en las MISMAS particiones de confirmación, semillas 40–44):
    final/resultados/final.csv               original, estándar, P sin reducir (2n)
    reduccion/resultados/reduccion.csv       P v1 (τ+ρ, fitness KNN)
    v2/resultados/confirmacion_v2.csv        P v2 (pesos dobles + fitness KNN+RL+árbol)
    final/resultados/comparativa.csv         transformaciones alternativas
    v2/resultados/cribado_v2.csv             cribado de variantes (semillas de EXPLORACIÓN 0–1)

Escribe v2/numeros.tex (macros con prefijo \\dos…), v2/tablas.tex y v2/figuras/. Los documentos de v2 cargan también
final/numeros.tex y final/tablas.tex para la ablación de la búsqueda de (λ, w), que es común a las dos versiones.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, os.path.join(RAIZ, "final"))
import analisis as A                                                # noqa: E402

FIG = os.path.join(AQUI, "figuras")
RES = os.path.join(AQUI, "resultados")
MODELOS, geo, resumen, holm = A.MODELOS, A.geo, A.resumen, A.holm
CLAVE = {"Regresión lineal": "RL", "KNN": "KNN", "Árbol": "Arbol", "Random Forest": "RF"}
ETIQ = {"estándar": "estándar", "P sin reducir": "P sin reducir (2n)", "P v1": "P v1 (fitness KNN, < n)",
        "P": "P v2 (< n)"}


def guardar(fig, nombre):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, nombre + ".pdf"))
    fig.savefig(os.path.join(FIG, nombre + ".png"), dpi=200)


# ------------------------------------------------------------------------------------------------------------ datos
def cargar_v2():
    r = pd.read_csv(os.path.join(RES, "confirmacion_v2.csv"))
    r = r[r.representacion == "multi"].copy()
    r["representacion"] = "P"
    return r


def cargar():
    """final.csv + P v1 + P v2, con rel (frente a original) y rel_est (frente a estándar)."""
    d = A.cargar_final()
    d = d[d.representacion != "P sin síntesis"].replace({"representacion": {"P": "P v1"}})
    r = cargar_v2()
    info = d.drop_duplicates(["dataset", "semilla"])[["dataset", "semilla", "n_filas", "n_atributos"]]
    r = r.drop(columns=["n"]).merge(info, on=["dataset", "semilla"])
    d = pd.concat([d.drop(columns=["rel", "rel_est"]), r], ignore_index=True)
    clave = list(zip(d.dataset, d.semilla, d.modelo))
    for ref, col in [("original", "rel"), ("estándar", "rel_est")]:
        x = d[d.representacion == ref].set_index(["dataset", "semilla", "modelo"]).mae
        d[col] = d.mae.values / x.loc[clave].values
    return d


def cargar_comparativa():
    c = A.cargar_comparativa()
    c = c[c.representacion != "P sin síntesis"].replace({"representacion": {"P": "P v1"}})
    r = cargar_v2().rename(columns={"dim": "n_atributos_salida", "segundos": "segundos_transf"})
    c = pd.concat([c, r[["dataset", "semilla", "representacion", "modelo", "mae", "segundos_transf",
                         "n_atributos_salida"]]], ignore_index=True)
    n = c[c.representacion == "original"].drop_duplicates(["dataset", "semilla"]).set_index(["dataset", "semilla"])
    c["frac_dim"] = c.n_atributos_salida.values / n.n_atributos_salida.loc[list(zip(c.dataset, c.semilla))].values
    ref = c[c.representacion == "original"].set_index(["dataset", "semilla", "modelo"]).mae
    c["rel"] = c.mae.values / ref.loc[list(zip(c.dataset, c.semilla, c.modelo))].values
    return c


# ------------------------------------------------------------------------------------------------------------ tablas
def tabla_principal(d):
    filas = []
    for m in MODELOS:
        s = d[(d.modelo == m) & (d.representacion == "P")]
        a, b = resumen(s), resumen(s.assign(rel=s.rel_est))
        fila = dict(modelo=m, **{k + "_orig": v for k, v in a.items()}, **{k + "_est": v for k, v in b.items()})
        for rep in ["estándar", "P sin reducir", "P v1"]:
            fila[rep] = geo(d[(d.modelo == m) & (d.representacion == rep)].groupby("dataset").rel.apply(geo))
        filas.append(fila)
    t = pd.DataFrame(filas).set_index("modelo")
    aj = holm(np.r_[t.p_orig.values, t.p_est.values])
    t["p_holm_orig"], t["p_holm_est"] = aj[:4], aj[4:]
    return t


def efecto(d, ref):
    filas = []
    for m in MODELOS:
        s = d[d.modelo == m]
        a = s[s.representacion == "P"].set_index(["dataset", "semilla"]).mae
        b = s[s.representacion == ref].set_index(["dataset", "semilla"]).mae
        com = a.index.intersection(b.index)
        filas.append(dict(modelo=m, **resumen((a[com] / b[com]).rename("rel").reset_index())))
    t = pd.DataFrame(filas).set_index("modelo")
    t["p_holm"] = holm(t.p.values)
    return t


def tabla_dimension(d):
    u = d[d.representacion == "P"].drop_duplicates(["dataset", "semilla"])
    t = u.groupby("dataset").agg(n=("n_atributos", "first"), orig=("n_orig", "mean"), comp=("n_comp", "mean"),
                                 dim=("dim", "mean"))
    return t.loc[[x for x in d.dataset.unique() if x in t.index]]


ORDEN = ["estándar", "YJ máx. verosimilitud", "cuantiles", "PCA 95 %", "PCA v-CV", "estándar + PCA", "MLKR",
         "PG (gplearn)", "P sin reducir", "P v1", "P"]


def tabla_comparativa(c):
    A.ORDEN_COMP = ORDEN
    return A.tabla_comparativa(c)


def tabla_cribado():
    cr = pd.read_csv(os.path.join(RES, "cribado_v2.csv"))
    p = cr.pivot_table(index=["dataset", "semilla", "modelo"], columns="representacion", values="mae")
    por_ds = np.log(p.div(p["original"], axis=0)).groupby(["dataset", "modelo"]).mean()
    reps = [r for r in ["actual", "multi", "multi+parada", "multi+parada30", "multi+selección", "knn+parada"]
            if r in p.columns]
    g = np.exp(por_ds.groupby("modelo")[reps].mean()).T[MODELOS]
    g["media"] = np.exp(por_ds[reps].mean())
    u = cr[cr.representacion.isin(reps)].drop_duplicates(["dataset", "semilla", "representacion"])
    g["evaluaciones"] = u.groupby("representacion").evaluaciones.mean()
    g["segundos"] = u.groupby("representacion").segundos.mean()
    g["dim/n"] = (u.dim / u.n).groupby(u.representacion).mean()
    return g, cr[["dataset", "semilla"]].drop_duplicates().shape[0]


# ------------------------------------------------------------------------------------------------------------ figuras
def _barras(ax, d, fs=8, titulo=True):
    reps = [("estándar", A.GRIS), ("P v1", A.NARANJA), ("P", A.AZUL)]
    x, w = np.arange(4), 0.26
    for i, (rep, col) in enumerate(reps):
        vals = [geo(d[(d.modelo == m) & (d.representacion == rep)].groupby("dataset").rel.apply(geo)) for m in MODELOS]
        ax.bar(x + (i - 1) * w, np.array(vals) - 1, w * 0.92, bottom=1, color=col, label=ETIQ[rep])
        if rep == "P":
            for xi, v in zip(x, vals):
                ax.text(xi + w, v - 0.012 if v < 1 else v + 0.004, f"{100 * (v - 1):+.1f} %".replace(".", ","),
                        ha="center", va="top" if v < 1 else "bottom", fontsize=fs)
    ax.axhline(1, color=A.TINTA2, lw=1)
    ax.set_xticks(x, ["Regresión\nlineal", "KNN", "Árbol", "Random\nForest"])
    ax.set_ylabel("MAE / MAE con X original  (<1 mejora)")
    ax.set_ylim(0.7, 1.04)
    ax.legend(loc="lower left", fontsize=fs)
    if titulo:
        ax.set_title("e'/e por modelo (media geométrica de los 12 datasets)", loc="left", fontsize=9)


def _heatmap(ax, d, fs=7.5, titulo=True):
    s = d[d.representacion == "P"]
    t = s.groupby(["dataset", "modelo"]).rel.apply(geo).unstack()[MODELOS]
    t = t.loc[t.mean(axis=1).sort_values().index]
    lim = max(abs(np.log(t.values)).max(), 0.05)
    ax.imshow(np.log(t.values), cmap=A.DIVERGENTE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(4), ["R. lineal", "KNN", "Árbol", "RF"])
    ax.set_yticks(range(len(t)), [A.corto(i) for i in t.index])
    ax.grid(False)
    for i in range(t.shape[0]):
        for j in range(4):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=fs,
                    color="white" if abs(np.log(v)) > 0.6 * lim else A.TINTA)
    if titulo:
        ax.set_title("e'/e de P v2 por dataset", loc="left", fontsize=9)


def fig_principal(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.05, 1]))
    _barras(a1, d)
    _heatmap(a2, d)
    fig.tight_layout()
    return fig


def fig_rf(c, modelo="Random Forest", figsize=(5.4, 3.0)):
    reps = ["PCA 95 %", "estándar + PCA", "P sin reducir", "P v1", "P"]
    txt = {"PCA 95 %": "solo PCA (95 %, < n)", "estándar + PCA": "X estándar + su PCA (2n)",
           "P sin reducir": "P sin reducir (2n)", "P v1": "P v1: fitness KNN (< n)", "P": "P v2: KNN+RL+árbol (< n)"}
    c, _ = A._completos(c, reps + ["original"])
    por_ds = c[c.modelo == modelo].groupby(["dataset", "representacion"]).rel.apply(geo).unstack()
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    for i, r in enumerate(reps):
        v = por_ds[r].dropna()
        g = geo(v)
        col = A.AZUL if r == "P" else (A.NARANJA if r.startswith("P ") else A.GRIS)
        ax.barh(i, g - 1, left=1, color=col, height=0.6)
        ax.scatter(np.clip(v, 0.8, 1.3), i + rng.uniform(-0.18, 0.18, len(v)), s=9, color=A.TINTA2, zorder=3, lw=0)
        ax.text(1.43, i, f"{100 * (g - 1):+.1f} %".replace(".", ","), va="center", ha="right", fontsize=8,
                fontweight="bold" if r == "P" else "normal")
    ax.axvline(1, color=A.TINTA2, lw=1)
    ax.axvspan(1.31, 1.44, color="white", zorder=2)
    ax.set_xlim(0.8, 1.44)
    ax.set_yticks(range(len(reps)), [txt[r] for r in reps], fontsize=8)
    ax.set_xlabel(f"MAE / MAE con X original ({modelo})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_cribado(g):
    reps = list(g.index)
    txt = {"actual": "P v1 (fitness KNN)", "multi": "fitness KNN+RL+árbol", "multi+parada": "+ parada (paciencia 15)",
           "multi+parada30": "+ parada (paciencia 30)", "multi+selección": "+ elegir θ con apartado",
           "knn+parada": "pesos dobles, fitness KNN + parada"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]), sharey=True)
    y = np.arange(len(reps))
    a1.barh(y, g["media"] - 1, left=1, color=[A.AZUL if r == "multi" else A.GRIS for r in reps], height=0.6)
    for yi, v in zip(y, g["media"]):
        a1.text(v - 0.002, yi, f"{v:.3f}".replace(".", ","), va="center", ha="right", fontsize=8, color=A.TINTA)
    a1.axvline(1, color=A.TINTA2, lw=1)
    a1.set_xlim(0.85, 1.0)
    a1.set_yticks(y, [txt[r] for r in reps], fontsize=8)
    a1.set_xlabel("e'/e medio de los 4 modelos")
    a1.invert_yaxis()
    a2.barh(y, g["segundos"], color=[A.AZUL if r == "multi" else A.GRIS for r in reps], height=0.6)
    a2.set_xlabel("segundos por búsqueda")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------------------ LaTeX
_num, _pct, _p = A._num, A._pct, A._p


def exportar(d, c):
    m, tablas = {}, {}
    tp = tabla_principal(d)
    filas = []
    for mod, k in CLAVE.items():
        r = tp.loc[mod]
        m[f"dosRel{k}"], m[f"dosCambio{k}"] = _num(r.cociente_orig), _pct(r.cambio_orig)
        m[f"dosDs{k}"], m[f"dosP{k}"] = f"{int(r.ds_mejora_orig)}/{int(r.ds_orig)}", _p(r.p_holm_orig)
        m[f"dosCambioVsEst{k}"], m[f"dosPVsEst{k}"] = _pct(r.cambio_est), _p(r.p_holm_est)
        m[f"dosPeorDs{k}"] = _pct(100 * (r.peor_ds_orig - 1))
        filas.append(f"{A.CORTO_MOD[mod]} & {_num(r['estándar'])} & {_num(r['P sin reducir'])} & {_num(r['P v1'])} & "
                     f"\\textbf{{{_num(r.cociente_orig)}}} & {int(r.ds_mejora_orig)}/{int(r.ds_orig)} & "
                     f"{_p(r.p_holm_orig)}\\\\\n")
    tablas["filasPrincipalDos"] = filas
    m["dosCambioMedio"] = _pct(100 * (geo(tp.cociente_orig) - 1))
    m["dosCambioMedioUno"] = _pct(100 * (geo(tp["P v1"]) - 1))
    for nombre, ref in [("VsUno", "P v1"), ("VsSr", "P sin reducir")]:
        ef = efecto(d, ref)
        for mod, k in CLAVE.items():
            m[f"dos{nombre}{k}"], m[f"dosP{nombre}{k}"] = _pct(ef.loc[mod, "cambio"]), _p(ef.loc[mod, "p_holm"])
            m[f"dosDs{nombre}{k}"] = f"{int(ef.loc[mod, 'ds_mejora'])}/{int(ef.loc[mod, 'ds'])}"
    # dimensión
    u = d[d.representacion == "P"].drop_duplicates(["dataset", "semilla"])
    m["dosDimMedia"] = _num(100 * (u.dim / u.n_atributos).mean(), 0) + "\\,\\%"
    m["dosDimMin"] = _num(100 * (u.dim / u.n_atributos).min(), 0) + "\\,\\%"
    m["dosOrigMedia"], m["dosCompMedia"] = _num(u.n_orig.mean(), 1), _num(u.n_comp.mean(), 1)
    m["dosFracConComp"] = _num(100 * (u.n_comp > 0).mean(), 0) + "\\,\\%"
    m["dosSegMediana"], m["dosSegMax"] = _num(u.segundos.median(), 0), _num(u.segundos.max(), 0)
    m["dosNCasos"] = str(len(u))
    td = tabla_dimension(d)
    tablas["filasDimensionDos"] = [f"{A.corto(i).replace('_', chr(92) + '_')} & {int(r.n)} & {_num(r.orig, 1)} & "
                                   f"{_num(r.comp, 1)} & {_num(r.dim, 1)}\\\\\n" for i, r in td.iterrows()]
    # máxima reducción
    s = d[d.representacion == "P"]
    g = s.groupby(["dataset", "modelo"])[["rel", "rel_est"]].agg(geo)
    esc = lambda x: A.corto(x).replace("_", "\\_")
    for mod, k in CLAVE.items():
        gm = g.xs(mod, level="modelo")
        m[f"dosMaxDs{k}"], m[f"dosMax{k}"] = esc(gm.rel.idxmin()), _pct(100 * (gm.rel.min() - 1))
    (ds_, mod_), r = g.rel.idxmin(), g.loc[g.rel.idxmin()]
    m["dosMaxDs"], m["dosMaxMod"] = esc(ds_), mod_
    m["dosMaxCambio"], m["dosMaxVsEst"] = _pct(100 * (r.rel - 1)), _pct(100 * (r.rel_est - 1))
    mae = s[(s.dataset == ds_) & (s.modelo == mod_)].mae.mean()
    mae_o = d[(d.dataset == ds_) & (d.modelo == mod_) & (d.representacion == "original")].mae.mean()
    mae_e = d[(d.dataset == ds_) & (d.modelo == mod_) & (d.representacion == "estándar")].mae.mean()
    m["dosMaxMae"], m["dosMaxMaeOrig"], m["dosMaxMaeEst"] = _num(mae, 3), _num(mae_o, 3), _num(mae_e, 3)
    # comparativa
    tc, ncomp = tabla_comparativa(c)
    nombres = {"estándar": "estándar", "YJ máx. verosimilitud": "YJ máx.\\ verosimilitud", "cuantiles": "cuantiles",
               "PCA 95 %": "solo PCA (95\\,\\%)", "PCA v-CV": "solo PCA (\\% por VC)",
               "estándar + PCA": "estándar + su PCA", "MLKR": "MLKR", "PG (gplearn)": "PG (gplearn)",
               "P sin reducir": "$P$ sin reducir", "P v1": "$P$ v1", "P": "\\textbf{$P$ v2}"}
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
    tablas["filasComparativaDos"] = filas
    for r, k in [("P", "P"), ("P v1", "Uno"), ("P sin reducir", "Sr"), ("PG (gplearn)", "PG"), ("MLKR", "MLKR"),
                 ("estándar + PCA", "EstPCA"), ("YJ máx. verosimilitud", "YJMV"), ("PCA 95 %", "PCA")]:
        m[f"dosRango{k}"] = _num(tc.loc[r, "rango medio"], 1)
        for mod, km in CLAVE.items():
            m[f"dosComp{k}{km}"] = _pct(100 * (tc.loc[r, mod] - 1))
    orden = tc["rango medio"].sort_values()
    m["dosPosicionP"] = str(int(list(orden.index).index("P") + 1))
    m["dosNReps"] = str(len(tc))
    # cribado
    gc, ncr = tabla_cribado()
    txt = {"actual": "$P$ v1 (fitness KNN)", "multi": "\\textbf{KNN+RL+árbol}", "multi+parada": "\\quad + parada (pac.\\ 15)",
           "multi+parada30": "\\quad + parada (pac.\\ 30)", "multi+selección": "\\quad + elegir $\\theta$ con apartado",
           "knn+parada": "pesos dobles, KNN + parada"}
    tablas["filasCribadoDos"] = [f"{txt[r]} & " + " & ".join(_num(gc.loc[r, mod]) for mod in MODELOS) +
                                 f" & {_num(gc.loc[r, 'media'], 3)} & {int(round(gc.loc[r, 'evaluaciones']))} & "
                                 f"{_num(gc.loc[r, 'segundos'], 0)}\\\\\n" for r in gc.index]
    m["dosCribCasos"] = str(ncr)
    for r, k in [("actual", "Uno"), ("multi", "Multi"), ("multi+parada", "Parada"), ("multi+parada30", "ParadaTreinta"),
                 ("multi+selección", "Seleccion")]:
        if r in gc.index:
            m[f"dosCrib{k}"] = _num(gc.loc[r, "media"], 3)
            m[f"dosCrib{k}Seg"] = _num(gc.loc[r, "segundos"], 0)
            m[f"dosCrib{k}Eval"] = str(int(round(gc.loc[r, "evaluaciones"])))
    with open(os.path.join(AQUI, "numeros.tex"), "w") as fh:
        fh.write("% Generado por v2/analisis_v2.py: NO editar a mano.\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(AQUI, "tablas.tex"), "w") as fh:
        fh.write("% Generado por v2/analisis_v2.py.\n")
        for k, filas in tablas.items():
            fh.write(f"\\newcommand{{\\{k}}}{{%\n" + "".join(filas) + "}\n")
    return m


def todas():
    A.todas()                                       # refresca también las cifras de final/ (ablación de la búsqueda)
    d, c = cargar(), cargar_comparativa()
    guardar(fig_principal(d), "principal_v2")
    f, a = plt.subplots(figsize=(4.6, 3.3)); _barras(a, d, titulo=False); f.tight_layout(); guardar(f, "poster_barras_v2")
    f, a = plt.subplots(figsize=(3.9, 3.9)); _heatmap(a, d, titulo=False); f.tight_layout(); guardar(f, "poster_heatmap_v2")
    guardar(fig_rf(c), "rf_v2")
    guardar(fig_rf(c, "Árbol"), "arbol_v2")
    gc, _ = tabla_cribado()
    guardar(fig_cribado(gc), "cribado_v2")
    A.ORDEN_COMP = ORDEN
    guardar(A.fig_comparativa_cd(c, modelos=["KNN", "Random Forest"], figsize=(6.0, 5.2), columnas=1), "poster_cd_v2")
    plt.close("all")
    return exportar(d, c)


if __name__ == "__main__":
    matplotlib.use("Agg")
    todas()
    print("figuras en", FIG, "; cifras en v2/numeros.tex y v2/tablas.tex")
