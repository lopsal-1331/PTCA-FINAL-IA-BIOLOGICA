"""Construye notebook_v2.ipynb (todo el proceso de P v2). Después:
    python -m nbconvert --to notebook --execute --inplace notebook_v2.ipynb"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Transformación evolutiva de atributos v2: aumentar, ponderar y reducir

**Práctica final — IA con inspiración biológica** · Máster en Ciencia y Tecnología, UC3M · Andrea López Salazar

Aprendemos con **CMA-ES**, viendo solo el entrenamiento, una transformación $X \to X'$ con **menos atributos que $X$** ($\dim X'<n$). La versión 2 cambia dos cosas respecto a la primera:

1. **Aumentar y luego reducir.** Se construyen $2n$ candidatos (los $n$ atributos con su forma Yeo-Johnson y sus $n$ componentes principales), cada uno con su propio peso ("pesos dobles"), y la evolución se queda con los de peso $\ge \tau$, como mucho $n-1$.
2. **Fitness de varios modelos.** La búsqueda minimiza la **media del MAE normalizado de KNN, regresión lineal y un árbol**, no solo el de KNN, para que la representación sirva a modelos de familias distintas.

**Resultados** (12 datasets × 5 particiones de confirmación, las mismas de la v1): $X'$ usa el **83 %** de los atributos y siempre menos de $n$, con componentes en el 93 % de las particiones. El error baja en **regresión lineal −14,6 %** (11/12 datasets, significativo; la v1 se quedaba en −4,4 %) y en **KNN −21,2 %** (9/12; la v1 llegaba a −24,3 %). En árbol y Random Forest **no cambia en media**. En media de los cuatro modelos, la v2 mejora un −9,5 % frente al −7,6 % de la v1, y es la **mejor transformación de las que no aumentan la dimensión**. La **parada por sobreajuste** con datos apartados se probó y se descartó: ahorra tiempo, pero empeora el error.
""")

md(r"""
## 0. Cómo ejecutarlo

**Instalación** (una vez, desde la carpeta `PTCA-FINAL/`):

```bash
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab v2/notebook_v2.ipynb
```

- La **primera vez** hace falta internet: diez de los doce datasets se descargan de OpenML (se guardan en `~/scikit_learn_data`).
- Por defecto el notebook **no repite los experimentos**: lee los CSV guardados y rehace el análisis, las figuras y las cifras de los documentos en unos segundos. Para repetir uno, pon su entrada de `RECALCULAR` a `True` (el CSV anterior se guarda como `.bak`).
- `N_JOBS` = procesos en paralelo (por defecto, todos los núcleos). Tiempos con 20 núcleos; con $k$ núcleos, multiplícalos por $\approx 20/k$.

| Experimento | Script | Qué produce | Tiempo (20 núcleos) |
|---|---|---|---|
| `final` | `final/experimento_final.py` | referencias (original, estándar) y $P$ sin reducir ($2n$) | ≈ 20 min |
| `reduccion` | `reduccion/experimento_reduccion.py` | $P$ v1 (fitness KNN), para comparar | ≈ 25 min |
| `cribado_v2` | `v2/cribado_v2.py` | diseño de v2: variantes en semillas de exploración 0–1 | ≈ 15 min |
| `confirmacion_v2` | `v2/experimento_v2.py` | **$P$ v2** en las particiones de confirmación | ≈ 20 min |
| `comparativa` | `final/experimento_comparativa.py` | 8 transformaciones alternativas (necesita `final`) | ≈ 8 min |
| `busqueda_*` | `final/experimento_busqueda.py <bloque>` | ablación de la búsqueda de $(\lambda,w)$ | 10–25 min cada uno |
| `parada` | `final/experimento_parada.py` | comportamiento de CMA-ES | ≈ 5 min |
""")

code(r"""
import os, sys, shutil, subprocess, inspect, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.precision", 3)

# ---------------------------------------------------------------- configuración
N_JOBS = os.cpu_count()
RECALCULAR = {"final": False, "reduccion": False, "cribado_v2": False, "confirmacion_v2": False, "comparativa": False,
              "busqueda_controles": False, "busqueda_rango": False, "busqueda_algoritmos": False,
              "busqueda_inicio": False, "parada": False}
PRESUPUESTO_DEMO = 300              # evaluaciones de CMA-ES en la demostración (el método usa 1000)
COMPILAR_LATEX = False              # True = recompilar memoria, paper y póster de v2 (requiere LaTeX)

V2 = os.path.abspath(".")
RAIZ = os.path.dirname(V2)
FINAL, REDUC = os.path.join(RAIZ, "final"), os.path.join(RAIZ, "reduccion")
sys.path[:0] = [V2, FINAL, REDUC]
import metodo_v2 as v2, metodo, analisis as A, analisis_v2 as A2
print("Python", sys.version.split()[0], "·", N_JOBS, "procesos")
""")

md(r"""
## 1. El problema

El enunciado pide un programa $P$ que, a partir **solo** de $(X_t, y_t)$:

- haga **síntesis** de atributos y **reducción de la dimensionalidad** ($\dim X' < n$), con un **criterio de eliminación que dependa del algoritmo evolutivo**;
- consiga que un modelo **arbitrario** $M$ entrenado con $X'_t$ tenga menor MAE en validación que con $X_t$ ($e'<e$).

Cada familia de modelos es sensible a cosas distintas: KNN a la geometría (escalas, atributos ruidosos), la regresión lineal a la forma de cada atributo y los árboles solo a las **direcciones** por las que pueden cortar (son invariantes a cualquier cambio monótono por atributo).

**Qué falló en la v1.** La v1 (fitness solo de KNN) cumplía $\dim X'<n$ y mejoraba mucho a KNN, pero no a los árboles: como un árbol no cambia con la forma ni con los pesos, solo le ayudan las componentes (cortes oblicuos), y el fitness de KNN no tenía motivo para guardarles sitio en el presupuesto de $n-1$ atributos.
""")

md(r"""
## 2. El método (P v2)

![Esquema de P v2](figuras/esquema_v2.png)

**Genoma** $\theta = (\lambda_1..\lambda_n,\; v_1..v_{2n},\; \tau)$, con $3n+1$ números reales:

| Gen | Rango | Papel |
|---|---|---|
| $\lambda_j$ | $[-2, 2]$ | **forma**: exponente de Yeo-Johnson del atributo $j$ |
| $v_1..v_n$ | $[0, 3]$ | **peso** de cada atributo original (con su forma, estandarizado) |
| $v_{n+1}..v_{2n}$ | $[0, 3]$ | **peso** de cada componente principal (estandarizada) |
| $\tau$ | $[0, 3]$ | **umbral de eliminación** |

**Decodificación** (cualquier $\theta$ da un $X'$ válido):

1. $\tilde x_j = (\psi(x_j;\lambda_j) - \mu_j)/\sigma_j$ (forma + estandarizar, con $\mu,\sigma$ del entrenamiento).
2. **Aumentar**: $c = [\,\tilde x_1..\tilde x_n \mid p_1..p_n\,]$, con $p$ las componentes principales de $\tilde x$ estandarizadas. Se calculan **antes** de ponderar, así no hay dependencia circular y es barato.
3. **Ponderar**: $v_k\,c_k$.
4. **Reducir**: se conservan los $k$ con $v_k \ge \tau$; si son $n$ o más, los $n-1$ de mayor peso; si no queda ninguno, el de mayor peso. $\dim X' \le n-1$ siempre.

**Fitness**: media de los MAE de validación cruzada (5 pliegues) de **KNN ($k=5$), regresión lineal y un árbol de decisión**, cada uno dividido por el MAE de predecir la mediana, en una submuestra de hasta 2000 filas. El árbol hace de sustituto barato de Random Forest, que solo se mide al final. **Búsqueda**: CMA-ES, 1000 evaluaciones, inicio en $\theta_0$ = estandarizar sin componentes ($\lambda=1$, $v_{orig}=1$, $v_{comp}=0$, $\tau=0{,}5$).

**Por qué el criterio de eliminación es evolutivo**: CMA-ES decide qué candidatos sobreviven (originales o componentes, por sus pesos) y cuántos (por $\tau$). La restricción $\dim X'<n$ está en la decodificación.
""")

code(r"""
print(inspect.getsource(v2.TransformacionV2))
print(inspect.getsource(v2.hacer_fitness))
""")

md(r"""
## 3. Demostración en vivo

$P$ v2 sobre una partición de los dos datasets de la práctica con `PRESUPUESTO_DEMO` evaluaciones. $P$ solo ve el entrenamiento.
""")

code(r"""
from sklearn.model_selection import train_test_split
from experimento_final import modelos

SEMILLA = 40
demo, filas = {}, []
for nombre in ["california", "diabetes"]:
    X, y = metodo.cargar(nombre)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=SEMILLA)
    t0 = time.time()
    P = v2.aprender(Xt, yt, SEMILLA, parada=False, presupuesto=PRESUPUESTO_DEMO)     # <- P: solo entrenamiento
    demo[nombre] = (P, Xt)
    for mod, m in modelos(SEMILLA).items():
        e = np.mean(np.abs(m.fit(Xt, yt).predict(Xv) - yv))
        e2 = np.mean(np.abs(m.fit(P.transform(Xt), yt).predict(P.transform(Xv)) - yv))
        filas.append(dict(dataset=nombre, modelo=mod, e=e, **{"e'": e2}, cambio=f"{100*(e2/e-1):+.1f} %", cumple=e2 < e))
    print(f"{nombre}: n = {X.shape[1]} -> dim X' = {P.dim} ({P.n_orig} originales + {P.n_comp} componentes), "
          f"τ = {P.tau:.2f}; fitness {P.fitness:.3f}; {time.time()-t0:.0f} s")
pd.DataFrame(filas).set_index(["dataset", "modelo"])
""")

code(r"""
fig, ejes = plt.subplots(1, 2, figsize=(13, 3.6))
for ax, (nombre, (P, Xt)) in zip(ejes, demo.items()):
    cols = list(pd.read_csv(os.path.join(metodo.RAIZ, metodo.PRACTICA[nombre][0])).columns.drop(metodo.PRACTICA[nombre][1]))
    etiquetas = cols + [f"PC{i+1}" for i in range(len(cols))]
    dentro = np.isin(np.arange(2 * len(cols)), P.S)
    color = np.where(dentro, np.where(np.arange(2 * len(cols)) < len(cols), A.AZUL, A.VERDE), A.GRIS)
    ax.bar(range(len(etiquetas)), P.v, color=color)
    ax.axhline(P.tau, color=A.TINTA, ls="--", lw=1, label=f"τ = {P.tau:.2f}")
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=60, fontsize=7)
    ax.set_title(f"{nombre}: pesos dobles (azul: original conservado; verde: componente; gris: eliminado)", loc="left", fontsize=9)
    ax.legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
## 4. Protocolo

- **12 datasets** de regresión (los 2 de la práctica y 10 de OpenML; 6–12 atributos; ≤ 10 000 filas).
- **Diseño en dos fases.** Las variantes de v2 se compararon en un **cribado** con semillas de exploración (0–1) y 6 datasets. La elegida se **confirma** en las particiones 70/30 nuevas (semillas 40–44), las mismas de la v1, de modo que todas las comparaciones son pareadas.
- **Modelos triviales y fijos**: regresión lineal, KNN ($k=5$), árbol y Random Forest (100 árboles), por defecto.
- **Estadística por dataset** ($n=12$): media geométrica de $e'/e$ por dataset; Wilcoxon con Holm; Friedman con Nemenyi (Demšar, 2006).

## 5. Ejecución de los experimentos
""")

code(r"""
def ejecutar(clave, carpeta, script, csv, args=(), env_extra=None, borrar=True):
    ruta_csv = os.path.join(carpeta, "resultados", csv)
    if not RECALCULAR.get(clave, False):
        estado = "guardado" if os.path.exists(ruta_csv) else "NO EXISTE: pon RECALCULAR['%s'] = True" % clave
        print(f"[{clave}] se usa el CSV {estado}: {os.path.relpath(ruta_csv, RAIZ)}")
        return
    if borrar and os.path.exists(ruta_csv):
        shutil.copy(ruta_csv, ruta_csv + ".bak"); os.remove(ruta_csv)
    env = dict(os.environ, N_JOBS=str(N_JOBS), **(env_extra or {}))
    t0 = time.time()
    print(f"[{clave}] ejecutando {script} {' '.join(args)} ...", flush=True)
    r = subprocess.run([sys.executable, script, *args], cwd=carpeta, env=env, capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-1500:] if r.returncode else "")
    print(f"[{clave}] terminado en {(time.time()-t0)/60:.1f} min (código {r.returncode})")

ejecutar("final", FINAL, "experimento_final.py", "final.csv")
ejecutar("reduccion", REDUC, "experimento_reduccion.py", "reduccion.csv", env_extra={"VARIANTES": "τ+ρ α=0,τ α=0"}, borrar=False)
ejecutar("cribado_v2", V2, "cribado_v2.py", "cribado_v2.csv",
         env_extra={"VARIANTES": "actual,multi,multi+parada,multi+parada30,multi+selección,knn+parada"})
ejecutar("confirmacion_v2", V2, "experimento_v2.py", "confirmacion_v2.csv")
ejecutar("comparativa", FINAL, "experimento_comparativa.py", "comparativa.csv")
for bloque in ["controles", "rango", "algoritmos", "inicio"]:
    ejecutar("busqueda_" + bloque, FINAL, "experimento_busqueda.py", "busqueda.csv", args=(bloque,), borrar=False)
ejecutar("parada", FINAL, "experimento_parada.py", "parada.csv")
""")

md(r"""
## 6. Análisis completo

`analisis_v2.py` genera las figuras de `v2/figuras/` y las cifras de `v2/numeros.tex` y `v2/tablas.tex` (memoria, paper y póster de v2). También refresca `final/` (la ablación de la búsqueda es común).
""")

code(r"""
cifras = A2.todas()
d, c = A2.cargar(), A2.cargar_comparativa()
print(len(cifras), "cifras de v2 exportadas")
""")

md(r"""
## 7. Diseño de v2: el cribado

En semillas de **exploración** (0–1, 6 datasets), comparamos la v1 con las piezas nuevas. "Parada" aparta un 20 % de la submuestra de entrenamiento, evalúa ahí el mejor de cada generación y para si no mejora en 15 (o 30) generaciones; "elegir θ con apartado" no para, solo devuelve el mejor en esa parte apartada.
""")

code(r"""
gc, ncr = A2.tabla_cribado()
print(ncr, "particiones de exploración")
display(gc.round(3))
A2.fig_cribado(gc); plt.show()
""")

md(r"""
- **El fitness de tres modelos mejora la media** de los cuatro (0,887 frente a 0,912 de la v1): sobre todo en regresión lineal y el árbol, con algo de pérdida en KNN.
- **Parar por sobreajuste no ayuda.** Con paciencia de 15 generaciones para hacia las 440 evaluaciones (2,4 veces más rápido) pero empeora (0,902). Con más paciencia (0,897) o usando lo apartado solo para elegir el θ final, sin parar (0,895), tampoco se recupera. La pérdida no viene de cuándo se para, sino de quitar un 20 % de los datos al fitness para un criterio de 60–400 filas, demasiado ruidoso para decidir.
- **Elegida**: fitness KNN + regresión lineal + árbol, pesos dobles, sin parada.

## 8. Resultado en las particiones de confirmación
""")

code(r"""
t = A2.tabla_principal(d)
t[["estándar", "P sin reducir", "P v1", "cociente_orig", "ds_mejora_orig", "p_holm_orig", "cambio_est", "p_holm_est"]].rename(
    columns={"P sin reducir": "P sin reducir (2n)", "cociente_orig": "P v2 (e'/e)", "ds_mejora_orig": "datasets e'<e",
             "p_holm_orig": "p Holm", "cambio_est": "% vs estándar", "p_holm_est": "p Holm vs estándar"}).round(4)
""")

code(r"""
A2.fig_principal(d); plt.show()
display(Markdown("**P v2 frente a P v1** (< 1: v2 mejor)"))
display(A2.efecto(d, "P v1")[["cociente", "ds_mejora", "ds", "p", "p_holm"]].round(4))
display(Markdown("**P v2 frente a P sin reducir (2n)**"))
display(A2.efecto(d, "P sin reducir")[["cociente", "ds_mejora", "ds", "p", "p_holm"]].round(4))
""")

md(r"""
- **Regresión lineal**: −14,6 % (11/12 datasets, $p_{Holm}=0{,}027$), un 10,7 % mejor que la v1 ($p_{Holm}=0{,}010$). Elegir las componentes y la forma también para un modelo lineal le sirve, sobre todo en cpu_small (−60 %) y concrete (−33 %).
- **KNN**: −21,2 % (9/12), un 4 % peor que la v1 (no significativo) y ya **sin significación** tras Holm, por el caso negativo **kin8nm** (+28,5 %).
- **Árbol y Random Forest**: sin cambio en media (−0,6 % y +0,3 %). Ganan en algunos datasets (space_ga: RF −11,7 %, árbol −9,8 %; california: árbol −6 %) y pierden en otros (kin8nm, bank8FM).
- **Mayor reducción**: cpu_small con regresión lineal (−59,6 %; MAE de 6,25 a 2,52) y california con KNN (−58,7 %).

### Cuánto reduce y qué conserva
""")

code(r"""
display(A2.tabla_dimension(d).round(2))
u = d[d.representacion == "P"].drop_duplicates(["dataset", "semilla"])
print(f"dim X'/n medio: {(u.dim/u.n_atributos).mean():.2f} · originales {u.n_orig.mean():.1f} · componentes {u.n_comp.mean():.1f}"
      f" · particiones con componentes: {(u.n_comp>0).mean():.0%} · dim X' < n en todas: {bool((u.dim < u.n_atributos).all())}")
""")

md(r"""
- $X'$ conserva de media **4,9 atributos originales y 2,2 componentes**, con componentes en el **93 %** de las particiones (la v1, con fitness solo de KNN, sintetizaba en el 25 %). El fitness de varios modelos **sí reparte** el presupuesto de $n-1$ atributos entre originales y direcciones oblicuas.
- En **puma8NH**, donde pocos atributos explican $y$, se queda con 3 de 8.
- En **kin8nm** el reparto falla: todos los atributos importan por igual, y cambiar originales por componentes empeora a los cuatro modelos.

## 9. Comparativa con otras transformaciones
""")

code(r"""
tc, n = A2.tabla_comparativa(c)
print(n, "particiones; p: Wilcoxon por dataset de P v2 frente a cada una, con Holm")
display(tc[A.MODELOS + ["rango medio", "dim/n"]].round(3).sort_values("rango medio"))
A2.fig_rf(c); plt.show()
A2.fig_rf(c, "Árbol"); plt.show()
""")

md(r"""
- $P$ v2 es **segunda en rango medio** (4,5), solo por detrás de $P$ sin reducir (3,6), que usa $2n$ atributos, y la **mejor de las que no aumentan la dimensión**: supera a la v1 (5,3), a la programación genética (5,1, con $2n$), a añadir la PCA de $X$ estandarizado (5,2) y a MLKR (6,5).
- En **regresión lineal** es significativamente mejor que todas salvo la programación genética, con la que empata (0,85 frente a 0,84).
- **"Solo PCA"**, la otra forma de reducir, empeora a RF un 18 %: sustituir los ejes originales daña a los árboles; *elegir* cuáles conservar, no.

## 10. La búsqueda: piezas, inicio, rango y algoritmo

Estas ablaciones de la búsqueda de $(\lambda, w)$ se hicieron con $P$ sin reducir (fitness KNN). Siguen siendo válidas para la parte común de v2 (forma y pesos) y muestran que CMA-ES es robusto a su configuración.
""")

code(r"""
b = A.cargar_busqueda()
display(A.contra_referencia(b, [v for _, v, _ in A.ABLACION if v in set(b.variante)]).round(4))
ta, n = A.tabla_algoritmos(b); print(n, "particiones comunes"); display(ta.round(3))
""")

md(r"""
- Los **pesos** son la pieza principal (sin ellos, KNN empeora un 10 %); la búsqueda gana a la aleatoria (+6 %) y a los pesos por información mutua (+15 %).
- CMA-ES es robusto a la población inicial y al rango de $\lambda$; con el mismo presupuesto, tiene el mejor rango de fitness, aunque otras metaheurísticas (AG, HS, BBO) quedan cerca.

### Operadores evolutivos

| | CMA-ES (método) |
|---|---|
| Individuo | $\theta=(\lambda, v, \tau)$ real en $[0,1]^{3n+1}$, reescalado a los límites |
| Población inicial | $\mathcal N(\theta_0, 0{,}2^2 I)$, $\theta_0$ = estandarizar sin componentes |
| Tamaño | $\lambda = 4+\lfloor 3\ln D\rfloor$ |
| Selección | truncamiento por rango, $\mu=\lambda/2$, pesos logarítmicos |
| Recombinación | intermedia ponderada de los $\mu$ seleccionados |
| Mutación | $\mathcal N(m,\sigma^2C)$, $\sigma$ y $C$ adaptativas |
| Reemplazo | $(\mu/\mu_w,\lambda)$ generacional; se guarda el mejor evaluado |
| Restricciones | cubo $[0,1]^D$; $\dim X'<n$ por decodificación |
| Parada | 1000 evaluaciones (la parada por sobreajuste se probó y se descartó, sección 7) |

## 11. Coste y caso negativo
""")

code(r"""
print(f"Búsqueda de P v2: mediana {u.segundos.median():.0f} s, máximo {u.segundos.max():.0f} s por partición (20 procesos en paralelo)")
dia = d[(d.dataset == "diabetes") & (d.representacion.isin(["P v1", "P"]))].groupby(["representacion", "modelo"]).rel.apply(A.geo).unstack()
dia.round(3)
""")

md(r"""
Una búsqueda de v2 tarda ~5 minutos por partición con 20 procesos en paralelo (la v1, ~3 minutos): el árbol y la regresión lineal encarecen cada evaluación. Se paga una sola vez por dataset, y $X'$ tiene menos columnas que $X$. En **diabetes** (309 filas de entrenamiento) la v2 tampoco consigue trasladar la mejora del fitness a la validación.

## 12. Conclusiones

1. **$\dim X'<n$ siempre** (83 % de los atributos de media), con el criterio de eliminación aprendido por CMA-ES, y **síntesis real**: la v2 conserva componentes en el 93 % de las particiones.
2. **$e'<e$ en regresión lineal** (−14,6 %, significativo) **y KNN** (−21,2 %); en los árboles el error queda igual en media.
3. **El fitness decide para quién es la representación.** Con solo KNN, la v1 favorecía a KNN; con tres modelos, la v2 reparte mejor y gana sobre todo en regresión lineal (−9,5 % de media en los cuatro modelos, frente a −7,6 %).
4. **Es la mejor transformación de las que no aumentan la dimensión**, y segunda en rango medio entre once.
5. **La parada con datos apartados no compensa** con datasets de este tamaño.
6. **Límites**: 12 datasets pequeños; casos negativos (kin8nm, diabetes); ~5 minutos por búsqueda; a los árboles les ayuda sobre todo *añadir* dimensiones, lo que el enunciado no permite.
""")

code(r"""
if COMPILAR_LATEX:
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "esquema_v2_pdf.tex"], cwd=os.path.join(V2, "figuras"))
    for carpeta, tex in [("memoria", "memoria_v2.tex"), ("paper", "paper_v2.tex"), ("poster", "poster_v2.tex")]:
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", tex], cwd=os.path.join(V2, carpeta),
                           capture_output=True, text=True)
        print(carpeta, "OK" if r.returncode == 0 else "ERROR")
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook_v2.ipynb"))
print("ok", len(C), "celdas")
