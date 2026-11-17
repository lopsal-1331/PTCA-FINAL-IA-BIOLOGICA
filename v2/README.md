# P v2 — Aumentar, ponderar y reducir

Segunda versión del programa P. Respecto a la v1 (`reduccion/`, fitness solo de KNN):

1. **Aumentar y luego reducir**: 2n candidatos (atributos con forma Yeo-Johnson + sus componentes principales), un
   peso por candidato ("pesos dobles") y se conservan los de peso ≥ τ, como mucho n−1. CMA-ES aprende λ, los 2n pesos
   y τ (3n+1 genes): el criterio de eliminación es evolutivo y dim X' < n para cualquier individuo.
2. **Fitness de varios modelos**: media del MAE normalizado (por el de la mediana) de KNN, regresión lineal y un árbol.
3. **Parada por sobreajuste** (20 % apartado + paciencia): probada en el cribado y **descartada** — ahorra tiempo pero
   empeora el error. Sigue disponible con `aprender(..., parada=True)`.

## Ficheros
| Fichero | Qué es |
|---|---|
| `metodo_v2.py` | El programa P v2 (transformación, fitness, búsqueda con parada opcional) |
| `cribado_v2.py` | Cribado de variantes en semillas de exploración 0–1, 6 datasets (`resultados/cribado_v2.csv`) |
| `experimento_v2.py` | Confirmación en las particiones 40–44 (`resultados/confirmacion_v2.csv`) |
| `analisis_v2.py` | Cifras (`numeros.tex`, macros `\dos…`), tablas (`tablas.tex`) y figuras (`figuras/`) |
| `notebook_v2.ipynb` | **Todo el proceso**, ejecutable en otro ordenador (ver su sección 0); lo genera `construir_notebook_v2.py` |
| `memoria/memoria_v2.pdf` | Memoria (3 caras) |
| `paper/paper_v2.pdf` | Paper LNCS |
| `poster/poster_v2.pdf` | Póster A0 |
| `esquema_v2.tex` | Esquema TikZ del método (y `figuras/esquema_v2.pdf|png`) |

Los documentos cargan también `../final/numeros.tex` y `../final/tablas.tex` para la ablación de la búsqueda de
(λ, w), común a v1 y v2, y comparan con los resultados de `final/` y `reduccion/` en las mismas particiones.

## Regenerar
```bash
cd v2
python cribado_v2.py                    # ≈ 15 min con 20 núcleos (N_JOBS=… para cambiarlo)
python experimento_v2.py                # ≈ 20 min
python analisis_v2.py                   # figuras y cifras
python construir_notebook_v2.py && python -m nbconvert --to notebook --execute --inplace notebook_v2.ipynb
(cd memoria && latexmk -pdf memoria_v2.tex); (cd paper && latexmk -pdf paper_v2.tex); (cd poster && latexmk -pdf poster_v2.tex)
```
