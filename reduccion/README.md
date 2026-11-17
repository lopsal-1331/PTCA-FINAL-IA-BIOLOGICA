# Reducción de dimensionalidad aprendida por CMA-ES (dim X' < n)

El enunciado exige síntesis **y** reducción de dimensionalidad, con un criterio de eliminación que dependa del
algoritmo evolutivo. El P anterior (`final/metodo.py`) producía X' = [z | p] con **2n** atributos. Aquí está la
versión final de P, que siempre devuelve menos de n atributos. `final/metodo.py` no se modifica: se importa.

## Ficheros
| Fichero | Qué hace |
|---|---|
| `metodo_reduccion.py` | **El método final P.** Genoma θ = (λ, w, τ, ρ): se eliminan los atributos con w_j < τ (si no se elimina ninguno, el de menor peso) y el hueco hasta n−1 se rellena en una fracción ρ con componentes principales de z_S. CMA-ES aprende τ y ρ junto con λ y w. |
| `experimento_reduccion.py` | Confirmación en las particiones de `final/` (12 datasets × semillas 40–44, 4 modelos). Variantes: `τ+ρ α=0` (= **P**), `τ α=0` (= P solo reducción, sin síntesis); `τ α=0.02` y `τ α=0.05` (penalización por dimensión, cribado parcial, no se usan). |
| `cribado_posthoc.py` | Contraste: reducción aplicada **después** de la evolución con una regla fija (umbral w<0,1; regla 1-SE; PCA truncada), reutilizando los θ de `final/resultados/final.csv`. |
| `resumen.py` | Resumen rápido de `resultados/reduccion.csv` frente a `final.csv`. |
| `resultados/` | `reduccion.csv`, `cribado_posthoc.csv`. |

## Uso
```bash
cd reduccion
VARIANTES="τ+ρ α=0,τ α=0" python experimento_reduccion.py   # ≈ 25 min por variante con 20 núcleos; se puede reanudar
python cribado_posthoc.py                                    # ≈ 3 min
N_JOBS=8 python ...                                          # nº de procesos (por defecto, todos los núcleos)
```
El análisis (`final/analisis.py`) lee `resultados/reduccion.csv`: `τ+ρ α=0` es el P de la memoria, el póster, el
paper y el notebook; el P anterior aparece como «P sin reducir (2n)».

Nota sobre `cribado_posthoc.py`: `final.csv` guarda θ con 3 decimales, así que P se reproduce con pequeñas
diferencias (árbol hasta 9 %, el resto ≤ 1,3 %). Dentro del cribado todas las representaciones usan el mismo θ.
