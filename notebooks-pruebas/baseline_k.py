"""Línea base justa para el método 'k aprendido': estándar con k elegido por CV (mismos datos que el fitness)."""
import os
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd, knn_lib as L
from sklearn.model_selection import train_test_split, KFold
F="resultados_knn.csv"; S="resultados_knn_k.csv"
r=pd.read_csv(F)[["dataset","semilla"]].drop_duplicates()
hechos=set()
if os.path.exists(S): d=pd.read_csv(S); hechos=set(zip(d.dataset,d.semilla))
filas=[]
for ds,s in r.itertuples(index=False):
    if (ds,s) in hechos: continue
    X,y=L.cargar(ds); Xt,Xv,yt,yv=train_test_split(X,y,test_size=.3,random_state=s)
    sub=np.random.default_rng(s).choice(len(yt),min(2000,len(yt)),replace=False)
    e=L.estandar(Xt); Zs,ys=e(Xt[sub]),yt[sub]
    plie=list(KFold(5,shuffle=True,random_state=s).split(Zs))
    cv={k:np.mean([np.mean(np.abs(L.knn_rapido(Zs[a],ys[a],Zs[b],k)-ys[b])) for a,b in plie]) for k in range(1,51)}
    k=min(cv,key=cv.get)
    filas.append(dict(dataset=ds,semilla=s,metodo="estándar + k por CV",mae=L.mae_knn(e(Xt),e(Xv),yt,yv,k),k_aprendido=k))
pd.DataFrame(filas).to_csv(S,mode="a",header=not os.path.exists(S),index=False); print(len(filas),"filas")
