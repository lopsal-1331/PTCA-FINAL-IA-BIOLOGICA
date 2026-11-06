import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import pandas as pd, lib
from joblib import Parallel, delayed
F="resultados_algoritmos.csv"
ALG=list(lib.ALGS)+["CMA-ES","búsqueda aleatoria","θ inicial (sin búsqueda)"]
trab=[(d,s,a) for s in [13,14] for d in ["diabetes","california"] for a in ALG]
n=0
for r in Parallel(n_jobs=18,return_as="generator_unordered")(delayed(lib.correr)(*t) for t in trab):
    pd.DataFrame([r]).to_csv(F,mode="a",header=not os.path.exists(F),index=False); n+=1; print(n,"/",len(trab),flush=True)
print("TERMINADO",flush=True)
