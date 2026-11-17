import pandas as pd, numpy as np
from scipy.stats import wilcoxon
d = pd.read_csv("resultados_componente3.csv")
b = d[d.representacion == "estándar"].set_index(["dataset", "semilla", "modelo"]).MAE
d["rel"] = d.apply(lambda r: r.MAE / b[(r.dataset, r.semilla, r.modelo)], axis=1)
pd.set_option("display.width", 200)
print("MAE relativo medio vs estándar (<1 mejora):"); print(d.pivot_table(index="modelo", columns="representacion", values="rel").round(4), "\n")
print("Mediana entre datasets:"); print(d.groupby(["modelo", "representacion"]).apply(lambda g: g.groupby("dataset").rel.mean().median()).unstack().round(4), "\n")
for rep in ["puerta", "YJ+pesos+puerta"]:
    print(f"Por dataset, MAE rel. de '{rep}':"); print(d[d.representacion == rep].pivot_table(index="dataset", columns="modelo", values="rel").round(3), "\n")
print("PCA seleccionadas / n_atributos (media):"); print(d[d.modelo == "Árbol"].groupby(["representacion"]).apply(lambda g: (g.n_pca_sel / g.n_atributos).mean()).round(2), "\n")
print("Mejora vs estándar (datasets/12), Wilcoxon p; y 'puerta' vs 'todas':")
for mod in d.modelo.unique():
    for rep in ["estándar+PCA (todas)", "puerta", "YJ+pesos+puerta"]:
        v = d[(d.modelo == mod) & (d.representacion == rep)].groupby("dataset").rel.mean()
        print(f"{mod:14s} {rep:21s} {int((v<1).sum()):2d}/12 p={wilcoxon(v-1).pvalue:.3f}")
