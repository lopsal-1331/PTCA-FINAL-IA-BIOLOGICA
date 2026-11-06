import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import pandas as pd, pca_lib as P, llr_exp as E
from joblib import Parallel, delayed
F="resultados_llr.csv"
trab=[(d,s) for s in range(30,36) for d in P.DATASETS[::-1]]
n=0
for fs in Parallel(n_jobs=18,return_as="generator_unordered")(delayed(E.correr)(d,s) for d,s in trab):
    pd.DataFrame(fs).to_csv(F,mode="a",header=not os.path.exists(F),index=False); n+=1; print(n,"/",len(trab),flush=True)
print("TERMINADO",flush=True)
