import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import pandas as pd, hibrido as H
from joblib import Parallel, delayed
F="resultados_hibrido.csv"
trab=[(d,s,v) for s in range(13,19) for d in ["diabetes","california"] for v in H.VARIANTES]
n=0
for r in Parallel(n_jobs=18,return_as="generator_unordered")(delayed(H.correr)(*t) for t in trab):
    pd.DataFrame([r]).to_csv(F,mode="a",header=not os.path.exists(F),index=False); n+=1; print(n,"/",len(trab),flush=True)
print("TERMINADO",flush=True)
