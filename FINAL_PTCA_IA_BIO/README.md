# Práctica final — IA con inspiración biológica: *Aumentar, ponderar y reducir*

Programa **P** que aprende con **CMA-ES**, solo con el entrenamiento, una transformación `X → X'` con **menos atributos
que X** y que reduce el MAE de modelos de familias distintas.

1. **Forma**: Yeo-Johnson por atributo (λ ∈ [−2, 2]) y estandarizar.
2. **Aumentar (síntesis)**: 2n candidatos = los n atributos + sus n componentes principales.
3. **Ponderar**: un peso por candidato.
4. **Reducir**: se conservan los de peso ≥ τ, como mucho n − 1. **CMA-ES aprende τ** (criterio de eliminación
   evolutivo) y la restricción se cumple para cualquier individuo.

Fitness: media del MAE de validación cruzada de KNN, regresión lineal y un árbol (normalizado por el de la mediana).

## Contenido
| Fichero | Qué es |
|---|---|
| `notebook_final.ipynb` | **Todo el proceso**: requisitos, método, demostración, experimentos, resultados y justificación de cada decisión |
| `memoria/memoria.pdf` | Memoria (3 caras, doble columna) |
| `paper/paper.pdf` | Paper (Springer LNCS) |
| `poster/poster.pdf` | Póster A0 |
| `ptca.py` | El programa P (datos, transformación, fitness, CMA-ES y variantes) y la estadística |
| `experimentos.py` | Los cuatro experimentos: `cribado`, `confirmacion`, `ablacion`, `comparativa` |
| `alternativas.py` | Transformaciones con las que se compara P |
| `analisis.py` | Tablas, contrastes, figuras y cifras (`numeros.tex`, `tablas.tex`) de los documentos |
| `construir_notebook.py` | Genera el notebook |
| `esquema.tex` | Esquema del método (TikZ) |
| `datos/` | Los dos datasets de la práctica (los otros diez se descargan de OpenML) |
| `resultados/` | CSV de los experimentos |

## Ejecutar
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebook_final.ipynb                 # por defecto lee resultados/ y rehace el análisis en segundos
```
Repetir experimentos (con `N_JOBS=k` procesos; tiempos con 20):
```bash
python experimentos.py cribado        # ≈ 25 min  — diseño, semillas de exploración 0–1
python experimentos.py confirmacion   # ≈ 20 min  — método final, particiones 40–44
python experimentos.py ablacion       # ≈ 40 min  — una pieza cambiada cada vez, particiones 40–41
python experimentos.py comparativa    # ≈ 8 min   — 8 transformaciones alternativas
python analisis.py                    # figuras y cifras (numeros.tex, tablas.tex)
python construir_notebook.py && python -m nbconvert --to notebook --execute --inplace notebook_final.ipynb
(cd memoria && latexmk -pdf memoria.tex); (cd paper && latexmk -pdf paper.tex); (cd poster && latexmk -pdf poster.tex)
```
Ninguna cifra de los documentos está escrita a mano: todas vienen de `numeros.tex` y `tablas.tex`.
