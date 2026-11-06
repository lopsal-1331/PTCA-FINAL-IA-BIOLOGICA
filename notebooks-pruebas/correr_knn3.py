import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import pandas as pd, knn_lib2 as M
from joblib import Parallel, delayed
F="resultados_knn3.csv"
trab=[(d,s,m,v) for s in [13,14,15] for d in ["diabetes","california"] for m in ["mean","llr","res_llr"] for v in ["reg","comp","reg+comp"]]
n=0
for r in Parallel(n_jobs=18,return_as="generator_unordered")(delayed(M.trabajo2)(*t) for t in trab):
    pd.DataFrame([r]).to_csv(F,mode="a",header=not os.path.exists(F),index=False); n+=1; print(n,"/",len(trab),flush=True)
print("TERMINADO",flush=True)
