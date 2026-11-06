import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import pandas as pd, knn_lib as L
from joblib import Parallel, delayed
DS=["diabetes","california","concrete_compressive_strength","stock","space_ga","sulfur","kin8nm","bank8FM","puma8NH","house_8L","cpu_small","wine_quality"]
SEM=list(range(13,23)); F="resultados_knn.csv"
hechos=set()
if os.path.exists(F):
    d=pd.read_csv(F); hechos=set(zip(d.dataset,d.semilla))
trab=[(d,s) for s in SEM for d in DS if (d,s) not in hechos]
for d in DS: L.cargar(d)
def hacer(d,s): return L.un_trabajo(d,s)
n=0
for filas in Parallel(n_jobs=20, return_as="generator_unordered")(delayed(hacer)(d,s) for d,s in trab):
    pd.DataFrame(filas).to_csv(F,mode="a",header=not os.path.exists(F),index=False)
    n+=1; print(n,"/",len(trab),flush=True)
print("FIN",flush=True)
