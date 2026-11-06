import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np, pandas as pd, knn_lib as L, knn_lib2 as M
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, KFold
F="resultados_knn2.csv"; DS=["diabetes","california"]; SEM=[13,14,15]

def bases(ds,s):
    X,y=L.cargar(ds); Xt,Xv,yt,yv=train_test_split(X,y,test_size=.3,random_state=s)
    e=L.estandar(Xt); sub=np.random.default_rng(s).choice(len(yt),min(2000,len(yt)),replace=False)
    Zs,ys=e(Xt[sub]),yt[sub]; plie=list(KFold(5,shuffle=True,random_state=s).split(Zs))
    cv={k:np.mean([np.mean(np.abs(L.knn_rapido(Zs[a],ys[a],Zs[b],k)-ys[b])) for a,b in plie]) for k in range(1,51)}
    k=min(cv,key=cv.get)
    return [dict(dataset=ds,semilla=s,modo="base: estándar k=5",mae=L.mae_knn(e(Xt),e(Xv),yt,yv,5),k=5),
            dict(dataset=ds,semilla=s,modo="base: estándar + k por CV",mae=L.mae_knn(e(Xt),e(Xv),yt,yv,k),k=k),
            dict(dataset=ds,semilla=s,modo="base: crudo k=5",mae=L.mae_knn(Xt,Xv,yt,yv,5),k=5)]
filas=[r for d in DS for s in SEM for r in bases(d,s)]
pd.DataFrame(filas).to_csv(F,index=False)
trab=[(d,s,m) for s in SEM for d in DS for m in M.MODOS]
n=0
for r in Parallel(n_jobs=18,return_as="generator_unordered")(delayed(M.trabajo)(d,s,m) for d,s,m in trab):
    pd.DataFrame([r]).to_csv(F,mode="a",header=False,index=False); n+=1; print(n,"/",len(trab),flush=True)
print("FIN",flush=True)
