import pandas as pd, numpy as np
from scipy.stats import wilcoxon
d = pd.read_csv("resultados.csv")
# MAE relativo a "estándar" (misma semilla/dataset/modelo): <1 = mejora
b = d[d.representacion == "estándar"].set_index(["dataset", "semilla", "modelo"]).MAE
d["rel"] = d.apply(lambda r: r.MAE / b[(r.dataset, r.semilla, r.modelo)], axis=1)
pd.set_option("display.width", 200)
print("MAE relativo medio vs estándar (<1 mejora), media sobre datasets y semillas:")
print(d.pivot_table(index="modelo", columns="representacion", values="rel", aggfunc="mean").round(4), "\n")
print("Mediana por dataset (media sobre semillas) del MAE relativo:")
print(d.groupby(["modelo", "representacion"]).apply(lambda g: g.groupby("dataset").rel.mean().median()).unstack().round(4), "\n")
print("Por dataset (media de semillas), MAE rel. de PCA completa:")
print(d[d.representacion == "PCA completa"].pivot_table(index="dataset", columns="modelo", values="rel").round(3), "\n")
print("Por dataset, MAE rel. de estándar+PCA:")
print(d[d.representacion == "estándar+PCA"].pivot_table(index="dataset", columns="modelo", values="rel").round(3), "\n")
print("Datasets donde mejora (rel<1) / 12  y Wilcoxon p (por dataset):")
for mod in d.modelo.unique():
    for rep in ["PCA completa", "PCA95", "estándar+PCA"]:
        v = d[(d.modelo == mod) & (d.representacion == rep)].groupby("dataset").rel.mean()
        print(f"{mod:14s} {rep:13s} mejora {int((v<1).sum()):2d}/12  p={wilcoxon(v-1).pvalue:.3f}")
