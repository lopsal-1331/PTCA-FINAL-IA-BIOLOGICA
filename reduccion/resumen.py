"""resumen.py — MAE relativo al original (media geométrica) de las variantes con reducción frente a las de final.csv."""
import os
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
f = pd.read_csv(os.path.join(AQUI, "..", "final", "resultados", "final.csv"))
r = pd.read_csv(os.path.join(AQUI, "resultados", "reduccion.csv")).rename(columns={"variante": "representacion"})
d = pd.concat([f[["dataset", "semilla", "modelo", "representacion", "mae"]], r[["dataset", "semilla", "modelo", "representacion", "mae"]]])
p = d.pivot_table(index=["dataset", "semilla", "modelo"], columns="representacion", values="mae")
p = p.dropna(axis=1, thresh=int(0.9 * len(p)))
rel = np.log(p.div(p["original"], axis=0))
g = np.exp(rel.groupby("modelo").mean())
g.loc["MEDIA"] = np.exp(rel.mean())
print("MAE relativo al original (media geométrica, < 1 mejora):")
print(g.drop(columns="original").round(3).to_string())
u = r.drop_duplicates(["dataset", "semilla", "representacion"])
u["frac"] = u.dim / u.n
print("\nDimensión de X' / n (media) y casos:")
print(u.groupby("representacion").agg(frac=("frac", "mean"), dim=("dim", "mean"), m=("m", "mean"), tau=("tau", "median"),
                                      casos=("dim", "size")).round(3).to_string())
print(u.pivot_table(index="dataset", columns="representacion", values="dim").round(1).assign(n=u.groupby("dataset").n.first()).to_string())
