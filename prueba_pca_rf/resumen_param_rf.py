import pandas as pd, sys
from scipy.stats import wilcoxon
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "resultados_param_rf.csv"); pd.set_option("display.width", 220)
b = d[d.representacion == "estándar"].set_index(["dataset", "semilla"]).MAE
d["rel"] = [r.MAE / b[(r.dataset, r.semilla)] for r in d.itertuples()]
t = d.groupby(["dataset", "representacion"]).rel.mean().unstack()
print(t.round(3), "\n"); print(f"{'':24s} media  mediana mejora  p(Wilcoxon)")
for c in t.columns:
    if c != "estándar": print(f"{c:24s} {t[c].mean():.4f} {t[c].median():.4f} {int((t[c]<1).sum()):2d}/{len(t)}  {wilcoxon(t[c]-1).pvalue:.3f}")
