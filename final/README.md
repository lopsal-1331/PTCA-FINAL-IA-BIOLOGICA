# Entrega final — Transformación evolutiva de atributos

Aportación: un programa `P` que aprende con CMA-ES, solo con el entrenamiento, una transformación `X → X'` en cuatro
bloques — **forma** (Yeo-Johnson), **relevancia** (pesos), **reducción** (se eliminan los atributos con peso menor
que un umbral τ **aprendido por CMA-ES**) y **síntesis** (componentes principales del espacio aprendido, sin pasar de
n−1 atributos) — de modo que **dim X' < n** siempre, y que reduce el MAE de KNN, regresión lineal, árbol y Random
Forest en 12 datasets × 5 particiones nuevas.

## Entregables
| Fichero | Qué es |
|---|---|
| `notebook_final.ipynb` | **Todo el proceso**: método, demostración en vivo, ejecución (opcional) de todos los experimentos, análisis y conclusiones. Ejecutable en otro ordenador (ver su sección 0) |
| `memoria/memoria.pdf` | Memoria de 3 caras, doble columna, plantilla `uc3mmaster` |
| `paper/paper.pdf` | Versión de congreso (LNCS) |
| `poster/poster.pdf` | Póster A0 vertical (tikzposter) |

## Código
| Fichero | Qué hace |
|---|---|
| `../reduccion/metodo_reduccion.py` | **El programa `P` final**: genoma (λ, w, τ, ρ), decodificación con dim X' < n, fitness y búsqueda |
| `../reduccion/experimento_reduccion.py` | Confirmación de `P` (12 datasets × semillas 40–44, 4 modelos) |
| `../reduccion/cribado_posthoc.py` | Contraste: reducción con regla fija aplicada después de la evolución |
| `metodo.py` | Piezas comunes (Yeo-Johnson, KNN rápido, CMA-ES, carga de datos) y el `P` anterior sin reducir (2n atributos) |
| `experimento_final.py` | Referencias (original, estándar) y `P` sin reducir en las particiones de confirmación |
| `experimento_comparativa.py` | Transformaciones alternativas en las mismas particiones |
| `experimento_busqueda.py` | Ablación de la búsqueda de (λ, w): `controles`, `rango`, `algoritmos`, `inicio` (y `varianza`, antigua) |
| `experimento_parada.py` | Generaciones, tamaños de población y criterio de parada de CMA-ES |
| `compat.py` | Compatibilidad de metric-learn y gplearn con scikit-learn reciente |
| `experimento_rango_yj.py` | (antiguo, 500 evaluaciones) sustituido por `experimento_busqueda.py rango` |
| `analisis.py` | Tablas, contrastes por dataset (Wilcoxon + Holm, Friedman + Nemenyi) y figuras; escribe `numeros.tex` y `tablas.tex` |
| `construir_notebook.py` | Genera el notebook (luego se ejecuta con nbconvert) |
| `esquema_metodo.tex` | Esquema TikZ del método, compartido por los documentos |
| `resultados/` | CSV de todos los experimentos citados |

## Regenerar todo
Desde `PTCA-FINAL/`, con el entorno de `requirements.txt` (`N_JOBS` = nº de procesos; por defecto todos los núcleos):
```bash
cd final
python experimento_final.py                                     # solo si se quieren repetir (≈20 min, 20 núcleos)
(cd ../reduccion && VARIANTES="τ+ρ α=0,τ α=0" python experimento_reduccion.py && python cribado_posthoc.py)
python experimento_comparativa.py
for b in controles rango algoritmos inicio; do python experimento_busqueda.py $b; done
python experimento_parada.py
python analisis.py                                              # figuras + cifras para LaTeX
python construir_notebook.py && python -m nbconvert --to notebook --execute --inplace notebook_final.ipynb
(cd figuras && pdflatex esquema.tex)
(cd memoria && latexmk -pdf memoria.tex); (cd paper && latexmk -pdf paper.tex); (cd poster && latexmk -pdf poster.tex)
```
O, más sencillo, abrir `notebook_final.ipynb` y poner a `True` lo que se quiera repetir en `RECALCULAR`.
Ninguna cifra de los documentos está escrita a mano: todas vienen de `numeros.tex`, así que si se repite un
experimento basta con volver a ejecutar `analisis.py` y recompilar.
