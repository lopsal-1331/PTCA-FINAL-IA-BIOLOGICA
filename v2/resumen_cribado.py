"""resumen_cribado.py — e'/e (media geométrica por dataset y luego entre datasets) de cada variante del cribado v2."""
import os
import sys
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(AQUI, "resultados", sys.argv[1] if len(sys.argv) > 1 else "cribado_v2.csv"))
p = d.pivot_table(index=["dataset", "semilla", "modelo"], columns="representacion", values="mae")
rel = np.log(p.div(p["original"], axis=0))
por_ds = rel.groupby(["dataset", "modelo"]).mean()
reps = [c for c in ["estándar", "actual", "multi", "multi+parada", "multi+parada30", "multi+selección", "knn+parada", "rel-media", "rel-max", "rel-mix"] if c in p.columns]
g = np.exp(por_ds.groupby("modelo")[reps].mean())
g.loc["MEDIA"] = np.exp(por_ds[reps].mean())
pd.set_option("display.width", 200)
print("e'/e frente a X original (media geométrica; < 1 mejora):")
print(g.round(3).to_string())
print("\nPor dataset, Random Forest:")
print(np.exp(por_ds.xs("Random Forest", level="modelo")[reps]).round(3).to_string())
u = d[~d.representacion.isin(["original", "estándar"])].drop_duplicates(["dataset", "semilla", "representacion"])
print("\nDimensión, componentes, evaluaciones y tiempo:")
print(u.assign(frac=u.dim / u.n).groupby("representacion")[["frac", "n_orig", "n_comp", "evaluaciones", "segundos"]]
      .mean().round(2).to_string())
