import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import pandas as pd, knn_lib3 as M
from joblib import Parallel, delayed
F="resultados_knn4.csv"
MET=["LLR-gauss, ancho CV (Powell)","LLR-gauss, pesos (CMA-ES)","LLR-gauss, YJ+pesos (CMA-ES)","LLR-gauss, pesos (gradiente)"]
trab=[(d,s,m) for s in [13,14,15] for d in ["diabetes","california"] for m in MET]
n=0
for r in Parallel(n_jobs=18,return_as="generator_unordered")(delayed(M.trabajo_est)(*t) for t in trab):
    pd.DataFrame([r]).to_csv(F,mode="a",header=not os.path.exists(F),index=False); n+=1; print(n,"/",len(trab),flush=True)
print("TERMINADO",flush=True)
