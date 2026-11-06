import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import pandas as pd, pca_lib as P
from joblib import Parallel, delayed
F="resultados_pca.csv"
MET=["crudo","estándar","estándar+PCA95","CMA-ES (YJ+pesos)","CMA-ES + PCA (peso por componente)","CMA-ES + PCA (corte suave)"]
for d in P.DATASETS: P.cargar(d)
trab=[(d,s,m) for s in [13,14,15] for d in P.DATASETS for m in MET[::-1]]
n=0
for r in Parallel(n_jobs=18,return_as="generator_unordered")(delayed(P.correr)(*t) for t in trab):
    pd.DataFrame([r]).to_csv(F,mode="a",header=not os.path.exists(F),index=False); n+=1; print(n,"/",len(trab),flush=True)
print("TERMINADO",flush=True)
