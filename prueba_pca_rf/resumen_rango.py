import pandas as pd, numpy as np
from scipy.stats import wilcoxon
d = pd.read_csv("resultados_rango_yj.csv"); pd.set_option("display.width", 200)
print("runs:", len(d))
for m in ["KNN", "KNN k-CV"]:
    rel = d.groupby(["dataset", "R"])[m].mean().unstack(); rel = rel.div(rel[2], axis=0)
    print(f"\n{m}: MAE relativo a R=2 (media de semillas), media/mediana entre datasets, nº datasets mejor que R=2, Wilcoxon p vs R=2")
    for R in rel.columns:
        v = rel[R]; print(f"R={R}: media {v.mean():.4f} mediana {v.median():.4f} mejor {int((v<1).sum())}/12 p={wilcoxon(v-1).pvalue if R!=2 else float('nan'):.3f}")
print("\nFitness de CV (train) medio relativo a R=2:"); f = d.groupby(["dataset", "R"]).fitness.mean().unstack(); print(f.div(f[2], axis=0).mean().round(4))
print("\nFracción de λ pegados al borde (media):"); print(d[d.R > 0].groupby("R").frac_borde.mean().round(3))
print("\n|λ| máximo aprendido (mediana entre runs):"); print(d[d.R > 0].groupby("R").lam_abs_max.median().round(2))
print("\nMAE KNN por dataset y R (media de semillas), relativo a R=2:")
print(d.groupby(["dataset", "R"]).KNN.mean().unstack().pipe(lambda t: t.div(t[2], axis=0)).round(3))
m = pd.read_csv("lambdas_mle.csv"); print("\nλ MLE (scipy): cuantiles", m.lam_mle.quantile([0, .05, .25, .5, .75, .95, 1]).round(2).to_dict())
print("fracción con |λ_MLE|>1:", round((m.lam_mle.abs() > 1).mean(), 3), " >2:", round((m.lam_mle.abs() > 2).mean(), 3), " >4:", round((m.lam_mle.abs() > 4).mean(), 3))
print("\nλ aprendido (todos los R=8): cuantiles", pd.Series(np.concatenate([np.array(s.split("|"), float) for s in d[d.R == 8].lam])).quantile([0, .05, .25, .5, .75, .95, 1]).round(2).to_dict())
