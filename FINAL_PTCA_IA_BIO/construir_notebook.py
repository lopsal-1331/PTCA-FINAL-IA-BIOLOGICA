"""Genera notebook_final.ipynb. Después: python -m nbconvert --to notebook --execute --inplace notebook_final.ipynb

Las cifras del texto se escriben como [[macro]] y se toman de numeros.tex (generado por analisis.py), igual que en la
memoria, el paper y el póster: ninguna está escrita a mano. Orden: analisis.py -> construir_notebook.py -> nbconvert."""
import os
import re
import nbformat as nbf

CARPETA = os.path.dirname(os.path.abspath(__file__))


def leer_numeros():
    """numeros.tex -> {macro: texto en Markdown} ($-$14{,}6\\,\\% -> −14,6 %)."""
    with open(os.path.join(CARPETA, "numeros.tex")) as fh:
        crudo = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", fh.read(), flags=re.M))
    traducir = [("$-$", "−"), ("$+$", "+"), ("{,}", ","), ("\\,\\%", " %"), ("\\%", "%"), ("\\_", "_"), ("$", "")]
    for a, b in traducir:
        crudo = {k: v.replace(a, b) for k, v in crudo.items()}
    return crudo


NUM = leer_numeros()


def cifras(s):
    faltan = sorted(set(re.findall(r"\[\[(\w+)\]\]", s)) - set(NUM))
    if faltan:
        raise KeyError(f"macros que no están en numeros.tex: {faltan}")
    return re.sub(r"\[\[(\w+)\]\]", lambda g: NUM[g.group(1)], s)


C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(cifras(s.strip("\n"))))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Aumentar, ponderar y reducir: transformación evolutiva de atributos

**Práctica final — IA con inspiración biológica** · Máster en Ciencia y Tecnología, UC3M · Andrea López Salazar

**Resumen.** $P$ aprende con **CMA-ES**, viendo solo el entrenamiento, una transformación $X\to X'$ con **menos atributos que $X$**: corrige la forma de cada atributo (Yeo-Johnson), **aumenta** el espacio con sus componentes principales, **pondera** cada uno de los $2n$ candidatos y **conserva** los de peso $\ge\tau$, como mucho $n-1$. El umbral $\tau$ (el criterio de eliminación) también lo aprende la evolución. En [[nDatasets]] datasets × [[nParticiones]] particiones no usadas para diseñarlo, $X'$ usa el [[dimMedia]] de los atributos (siempre menos de $n$) y el MAE baja en **regresión lineal** ([[cambioRL]], $p$ = [[pRL]]) y **KNN** ([[cambioKNN]]); en árbol y Random Forest queda igual. Tiene el mejor rango medio frente a ocho transformaciones conocidas, y cada decisión del diseño se contrasta con una alternativa.

| Sección | Contenido |
|---|---|
| 0 | Cómo ejecutarlo |
| 1 | Qué pide el enunciado y cómo lo cumple $P$ |
| 2 | El método (código) |
| 3 | Demostración en vivo |
| 4 | Protocolo experimental |
| 5 | Experimentos |
| 6 | Resultado: ¿$e'<e$ con $\dim X'<n$? |
| 7 | Justificación de cada decisión (cribado y ablación) |
| 8 | Comparativa con otras transformaciones |
| 9 | Conclusiones y límites |
""")

md(r"""
## 0. Cómo ejecutarlo

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebook_final.ipynb
```

- La **primera vez** hace falta internet: diez de los doce datasets se descargan de OpenML (se guardan en `~/scikit_learn_data`). Los dos de la práctica están en `datos/`.
- Por defecto se **leen los resultados guardados** en `resultados/` y se rehace todo el análisis en segundos. Para repetir un experimento, pon su entrada de `RECALCULAR` a `True` (el CSV anterior se guarda como `.bak`).
- `N_JOBS` = procesos en paralelo. Los tiempos son con 20 procesos; con $k$, multiplícalos por $\approx 20/k$.

| Experimento | Para qué | Particiones | Tiempo (20 procesos) |
|---|---|---|---|
| `cribado` | **elegir** el diseño (fitness, parada) | exploración: semillas 0–1, 7 datasets | ≈ 25 min |
| `confirmacion` | **medir** el método final | confirmación: semillas 40–44, 12 datasets | ≈ 20 min |
| `ablacion` | **justificar** cada pieza (cambiar una cada vez) | semillas 40–41, 12 datasets | ≈ 35 min |
| `comparativa` | **comparar** con 8 transformaciones conocidas | semillas 40–44, 12 datasets | ≈ 8 min |
""")

code(r"""
import os, sys, shutil, subprocess, inspect, time, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
warnings.filterwarnings("ignore"); pd.set_option("display.width", 200); pd.set_option("display.precision", 3)

N_JOBS = os.cpu_count()
RECALCULAR = {"cribado": False, "confirmacion": False, "ablacion": False, "comparativa": False}
import ptca, analisis as A
print(len(ptca.DATASETS), "datasets ·", N_JOBS, "procesos")
""")

md(r"""
## 1. Qué pide el enunciado y cómo lo cumple $P$

Un programa $P$ que, viendo **solo** $(X_t, y_t)$, aprenda $X \to X'$ de modo que un modelo **arbitrario** $M$ entrenado con $X'_t$ tenga menor MAE en validación ($e'<e$), empleando técnicas evolutivas para la **síntesis** de atributos y la **reducción de la dimensionalidad**.

| Requisito | Cómo se cumple | Dónde se comprueba |
|---|---|---|
| Solo con el entrenamiento | $P$ ajusta todo (forma, componentes, pesos, umbral) con $X_t$; la validación solo se usa para medir | `ptca.aprender`, `experimentos.tarea` |
| Computación evolutiva | CMA-ES busca $\theta=(\lambda, v, \tau)$ | sección 2; frente a búsqueda aleatoria y un AG en la sección 7 |
| Síntesis $z=f(x_1..x_n)$ | componentes principales del espacio con forma corregida como candidatos | sección 6 (cuántas se usan) y ablación "sin síntesis" |
| Reducción $s<n$ | como mucho $n-1$ atributos **para cualquier individuo** (en la decodificación) | sección 6: $\dim X'<n$ en todas las particiones |
| Criterio de eliminación evolutivo | CMA-ES aprende el umbral $\tau$ y los pesos: decide qué se elimina y cuánto | sección 2 |
| $e'<e$ para un $M$ arbitrario | una sola $X'$ para cuatro familias distintas (lineal, KNN, árbol, RF); fitness con tres de ellas | sección 6 |
""")

md(r"""
## 2. El método

![Esquema](figuras/esquema.png)

1. **Forma**: $\tilde x_j = (\psi(x_j;\lambda_j)-\mu_j)/\sigma_j$, Yeo-Johnson con $\lambda_j\in[-2,2]$ (corrige colas y asimetrías; admite negativos).
2. **Aumentar**: $2n$ candidatos $c=[\tilde x_1..\tilde x_n \mid p_1..p_n]$, con $p$ las componentes principales de $\tilde x$ (direcciones oblicuas: lo único que puede ayudar a un árbol, que es invariante a cambios monótonos de cada atributo).
3. **Ponderar**: un peso $v_k\in[0,3]$ por candidato (para KNN, una métrica aprendida).
4. **Reducir**: se conservan los de $v_k\ge\tau$; como mucho los $n-1$ de mayor peso.

**Fitness**: media del MAE de validación cruzada (5 pliegues) de KNN, regresión lineal y un árbol, cada uno dividido por el MAE de predecir la mediana (escala común, $f=1$ es no saber nada). **Búsqueda**: CMA-ES, 1000 evaluaciones, desde "estandarizar sin componentes".

| Operador | CMA-ES |
|---|---|
| Individuo | $\theta$ real en $[0,1]^{3n+1}$, reescalado a los límites |
| Población inicial | $\mathcal N(\theta_0, 0{,}2^2 I)$; $\lambda = 4+\lfloor 3\ln D\rfloor$ individuos por generación |
| Selección | truncamiento por **rango**: los $\mu=\lambda/2$ mejores (invariante a transformaciones monótonas del fitness; robusta al ruido) |
| Recombinación | intermedia ponderada de los $\mu$ seleccionados (promedia el ruido de la validación cruzada) |
| Mutación | $\mathcal N(m,\sigma^2C)$; $\sigma$ y $C$ adaptativas: aprende qué genes se mueven juntos |
| Reemplazo | $(\mu/\mu_w,\lambda)$ generacional; se guarda el mejor evaluado |
| Restricciones | cubo $[0,1]^D$; $\dim X'<n$ por decodificación |
| Parada | 1000 evaluaciones |
""")

code(r"""
print(inspect.getsource(ptca.Transformacion.fit))
print(inspect.getsource(ptca.hacer_fitness))
""")

md(r"""
## 3. Demostración en vivo

$P$ en una partición de california (300 evaluaciones para que sea rápido; el método usa 1000).
""")

code(r"""
Xt, Xv, yt, yv = ptca.particion("california", 40)
t0 = time.time()
P = ptca.aprender(Xt, yt, 40, presupuesto=300)                      # <- solo entrenamiento
print(f"n = {Xt.shape[1]} -> dim X' = {P.dim} ({P.n_orig} originales + {P.n_comp} componentes), τ = {P.tau:.2f}, {time.time()-t0:.0f} s")
filas = []
for m, mod in ptca.modelos(40).items():
    e = np.mean(np.abs(mod.fit(Xt, yt).predict(Xv) - yv))
    e2 = np.mean(np.abs(mod.fit(P.transform(Xt), yt).predict(P.transform(Xv)) - yv))
    filas.append(dict(modelo=m, e=e, **{"e'": e2, "cambio": f"{100*(e2/e-1):+.1f} %"}))
cols = list(pd.read_csv("datos/California.csv").columns.drop("MedHouseVal"))
etiq = cols + [f"PC{i+1}" for i in range(len(cols))]
fig, ax = plt.subplots(figsize=(10, 2.6))
dentro = np.isin(np.arange(2 * len(cols)), P.S)
ax.bar(range(len(etiq)), P.v, color=np.where(dentro, A.AZUL, A.GRIS)); ax.axhline(P.tau, color=A.TINTA, ls="--", label="τ")
ax.set_xticks(range(len(etiq)), etiq, rotation=60, fontsize=8); ax.legend(); ax.set_title("pesos de los 2n candidatos (azul: se conserva)", loc="left")
plt.show(); pd.DataFrame(filas).set_index("modelo")
""")

md(r"""
## 4. Protocolo experimental

- **12 datasets** de regresión: los 2 de la práctica y 10 de OpenML (6–12 atributos; ≤ 10 000 filas).
- **Diseño en dos fases**: las decisiones se toman en un **cribado** con semillas de *exploración* (0–1); el método elegido se mide en particiones 70/30 **nuevas** (semillas 40–44). Así el resultado final no está ajustado a las particiones con las que se mide.
- **Modelos triviales y fijos**: regresión lineal, KNN ($k=5$), árbol de decisión y Random Forest (100 árboles), con parámetros por defecto. El enunciado fija $M$: ajustarlo sería cambiar el modelo, no los datos.
- **Estadística por dataset**: las particiones de un dataset comparten filas, así que la unidad es el dataset ($n=12$). Cada dataset se resume con la media geométrica de $e'/e$; pares con **Wilcoxon** y corrección de **Holm**; comparaciones múltiples con **Friedman** y **Nemenyi** (Demšar, 2006).
- **Ningún dataset se excluye** a posteriori; el único caso negativo (kin8nm) se analiza aparte.

## 5. Experimentos
""")

code(r"""
def ejecutar(nombre):
    csv = os.path.join("resultados", nombre + ".csv")
    if not RECALCULAR[nombre]:
        print(f"[{nombre}] resultados guardados: {csv}" if os.path.exists(csv) else f"[{nombre}] FALTA {csv}"); return
    if os.path.exists(csv):
        shutil.copy(csv, csv + ".bak"); os.remove(csv)
    t0 = time.time()
    r = subprocess.run([sys.executable, "experimentos.py", nombre], env=dict(os.environ, N_JOBS=str(N_JOBS)),
                       capture_output=True, text=True)
    print(f"[{nombre}] {(time.time()-t0)/60:.1f} min", r.stderr[-1500:] if r.returncode else "")

for nombre in ["cribado", "confirmacion", "ablacion", "comparativa"]:
    ejecutar(nombre)
cifras = A.todas()                       # figuras/, numeros.tex y tablas.tex (memoria, paper y póster)
print(len(cifras), "cifras exportadas")
""")

md(r"""
## 6. Resultado: ¿$e'<e$ con $\dim X'<n$?

Confirmación: 12 datasets × 5 particiones nuevas. $e'/e$ (media geométrica), datasets en los que mejora y $p$ de Wilcoxon por dataset con Holm, frente a $X$ original y frente a $X$ estandarizado.
""")

code(r"""
t = A.tabla_principal()
display(t[["estándar", "cociente", "ds_mejora", "p_holm", "vs_est", "p_est_holm", "peor_ds", "peor"]].rename(columns={
    "cociente": "P (e'/e)", "ds_mejora": "datasets e'<e", "p_holm": "p Holm", "vs_est": "% vs estándar",
    "p_est_holm": "p Holm vs estándar", "peor_ds": "peor dataset", "peor": "e'/e peor"}))
A.fig_principal(); plt.show()
""")

md(r"""
**Lectura.** Con menos atributos que $X$, $P$ reduce el MAE de la **regresión lineal** un [[cambioRL]] (mejora en [[dsRL]] datasets; $p$ con Holm = [[pRL]]) y el de **KNN** un [[cambioKNN]] ([[dsKNN]] datasets). También mejora a estandarizar: RL [[vsEstRL]] ($p$ = [[pVsEstRL]]) y KNN [[vsEstKNN]]. Las mayores reducciones son [[maxRL]] en [[maxDsRL]] (RL; el MAE pasa de [[maxMaeOrig]] a [[maxMaeP]]) y [[maxKNN]] en [[maxDsKNN]] (KNN).

En el **árbol** ([[cambioArbol]]) y en **RF** ([[cambioRF]]) el error queda igual: un árbol no cambia con la forma ni con los pesos de un atributo, y con menos de $n$ atributos solo puede ganar si alguna dirección oblicua es mejor corte que el atributo original al que sustituye.

KNN no es significativo tras Holm ($p$ sin corregir = [[pSinHolmKNN]]) por un único dataset, **[[peorDsKNN]]** ([[peorKNN]]): sus atributos importan todos por igual y reducir obliga a perder información.
""")

code(r"""
display(Markdown("**Sensibilidad: sin kin8nm** (no es una selección: se informa del resultado con los 12)"))
display(A.tabla_sensibilidad()[["cambio", "ds_mejora", "ds", "p_holm"]])
td, u = A.tabla_dimension()
display(Markdown("**Reducción y síntesis por dataset** (media de 5 particiones)")); display(td.round(2))
print(f"dim X' < n en todas las particiones: {bool((u.dim < u.n).all())} · dim X'/n medio {(u.dim/u.n).mean():.2f} · "
      f"componentes en el {(u.n_comp > 0).mean():.0%} de las particiones · búsqueda: mediana {u.segundos.median():.0f} s")
""")

md(r"""
**Lectura.** Sin kin8nm (análisis de sensibilidad; el resultado que cuenta es el de los 12) KNN mejora [[sinKinKNN]] ($p$ = [[pSinKinKNN]]) y la media de los cuatro modelos pasa de [[cambioMedio]] a [[sinKinMedio]].

$\dim X'<n$ en **todas** las particiones: $X'$ usa de media el [[dimMedia]] de $n$, con [[origMedia]] atributos originales y [[compMedia]] componentes. La evolución elimina el [[fracElim]] de los atributos originales y conserva componentes en el [[fracComp]] de las particiones: **síntesis y reducción ocurren a la vez**, y qué se elimina lo deciden los pesos y el umbral $\tau$ que aprende CMA-ES. Cada búsqueda tarda una mediana de [[segMediana]] s (máximo [[segMax]] s).

## 7. Justificación de cada decisión

### 7.1 Cribado de diseño (semillas de exploración)
Cada fila cambia **cómo se evalúa** a los candidatos (fitness) o **cuándo se para**. "Un fitness por familia" da a cada modelo la $X'$ aprendida con su propio fitness (cuatro $X'$ distintas).
""")

code(r"""
gc, ncr = A.tabla_cribado(); print(ncr, "particiones de exploración"); display(gc.round(3))
A.fig_barras_tabla(gc, A.MODELOS, A.ET_CRIB, "P", "MAE / MAE con X original (exploración)"); plt.show()
""")

md(r"""
**Lectura.** *Fitness*: con solo KNN, la $X'$ sirve sobre todo a KNN ([[cribKnnKNN]]); con tres familias, la media de los cuatro modelos mejora ([[cribP]] frente a [[cribKnn]]). Exigir que mejoren todos a la vez (**minimax**: el peor modelo, cada uno relativo a sí mismo) empeora ([[cribMinimax]]): el máximo de cuatro estimaciones ruidosas premia el ruido. **Una $X'$ por familia** apenas mejora ([[cribFamilia]]) y deja de ser una sola $X'$ para un $M$ arbitrario, que es lo que pide el enunciado.

*Parada*: apartar un 20 % del entrenamiento para parar ante el sobreajuste ahorra tiempo ([[cribParadaSeg]] s frente a [[cribPSeg]] s), pero empeora ([[cribParada]]; [[cribParadaTreinta]] con más paciencia; [[cribSeleccion]] usando lo apartado solo para elegir $\theta$): quita datos al fitness a cambio de un criterio demasiado ruidoso. Por eso $P$ usa el fitness de tres familias y un presupuesto fijo de 1000 evaluaciones.

### 7.2 Ablación (particiones de confirmación)
Se cambia **una sola pieza** de $P$ cada vez. Cociente frente a $P$ (> 1: peor que $P$); $^*$ en la memoria: $p<0{,}05$ por dataset con Holm.
""")

code(r"""
ta, nab = A.tabla_ablacion(); print(nab, "particiones"); display(ta.round(3))
""")

md(r"""
**Lectura** ([[ablCasos]] particiones; cambio frente a $P$, > 0 es peor que $P$):

- **Forma** — la pieza que más aporta: sin ella ($\lambda=1$) RL empeora [[ablSinFormaRL]] (significativo).
- **Fitness de tres familias** — con solo KNN, KNN gana [[ablKnnKNN]] pero RL pierde [[ablKnnRL]] (significativo): una sola $X'$ para un $M$ arbitrario necesita varias familias en el fitness.
- **Búsqueda evolutiva** — con el mismo presupuesto (1000 evaluaciones), la búsqueda aleatoria empeora los cuatro modelos ([[ablAleat]] de media; RL [[ablAleatRL]], significativo).
- **CMA-ES frente a un AG** — empate ([[ablAG]]): cuenta buscar con selección y variación más que el algoritmo concreto. Se prefiere CMA-ES porque adapta sola la mutación, sin tasas de cruce y mutación que fijar a mano.
- **Límite de $n-1$** — sin él el error no cambia de forma apreciable ([[ablSinLimite]]), pero $X'$ crece hasta [[ablSinLimiteDim]]·$n$: el límite garantiza la reducción que pide el enunciado sin coste medible.
- **Síntesis** — sin ella el error tampoco cambia ([[ablSinSint]]). Las componentes no dan una mejora medible en estos datos, aunque la evolución las conserva en el [[fracComp]] de las particiones; se mantienen porque el enunciado pide síntesis y no empeoran nada.

## 8. Comparativa con otras transformaciones

Mismas particiones: estandarizar, Yeo-Johnson por máxima verosimilitud, cuantiles, "solo PCA" (95 % de varianza o % por validación cruzada), $X$ estandarizado + su PCA, MLKR (aprendizaje de métricas) y programación genética (`gplearn`). Rango medio de Friedman sobre los cuatro modelos y $\dim X'/n$.
""")

code(r"""
tc = A.tabla_comparativa()
display(tc[A.MODELOS + ["rango", "dim/n"] + ["p " + m for m in A.MODELOS]].sort_values("rango").round(3))
A.fig_cd(); plt.show()
""")

md(r"""
**Lectura.** $P$ tiene el **mejor rango medio** de Friedman ([[rangoP]]) entre nueve transformaciones, por delante de la programación genética ([[rangoPG]]), de $X$ estandarizado + su PCA ([[rangoEstPCA]]) y de Yeo-Johnson por máxima verosimilitud ([[rangoYJMV]]). Es la única que mejora a la vez RL y KNN **reduciendo** la dimensión:

- la programación genética empata en RL ([[compPGRL]]) pero usa $2n$ atributos;
- MLKR, que aprende una métrica para KNN, es la mejor en KNN ([[compMLKRKNN]]) pero apenas ayuda a RL ([[compMLKRRL]]);
- "solo PCA", la otra forma de reducir, empeora a RF un [[compPCARF]]: sustituir los ejes originales daña a los árboles; *elegir* cuáles conservar, no.

## 9. Conclusiones y límites

- **$P$ cumple todos los requisitos**: solo entrenamiento, búsqueda evolutiva, síntesis y reducción ($\dim X'<n$ siempre, [[dimMedia]] de $n$ de media) con un criterio de eliminación que aprende CMA-ES.
- **$e'<e$ en regresión lineal** ([[cambioRL]], significativo) **y KNN** ([[cambioKNN]]), con reducciones de hasta [[maxRL]] y [[maxKNN]]; en los árboles el error queda igual.
- **Mejor rango medio** entre nueve transformaciones, y la única que mejora RL y KNN a la vez reduciendo la dimensión.
- **Cada decisión está contrastada**: la forma, el fitness de tres familias y la búsqueda evolutiva son necesarias; el límite de $n-1$ garantiza la reducción sin coste; la parada por sobreajuste se probó y se descartó.
- **Límites**:
  - 12 datasets pequeños y numéricos, con un caso negativo ([[peorDsKNN]]).
  - La síntesis cumple el requisito pero no reduce el error de forma medible, y un AG rinde igual que CMA-ES.
  - Los árboles no mejoran porque lo que les ayuda es *añadir* dimensiones, que el enunciado no permite.
  - Cada búsqueda tarda unos [[segMediana]] s.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook_final.ipynb"))
print("ok", len(C), "celdas")
