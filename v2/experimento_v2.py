"""
experimento_v2.py — Confirmación de P v2 en las particiones de confirmación (12 datasets × semillas 40–44, 70/30),
las mismas que final/experimento_final.py y reduccion/experimento_reduccion.py, con los mismos 4 modelos.

P v2 = metodo_v2.aprender(modelos=KNN+RL+árbol, parada=False) — la variante elegida en el cribado (cribado_v2.py,
semillas de exploración 0–1). Las referencias (original, estándar, P sin reducir, P v1) están en los CSV de final/ y
reduccion/.

Uso: python experimento_v2.py      (≈15–20 min con 20 núcleos; se puede reanudar)
"""
import os
os.environ.setdefault("SEMILLAS", "40,41,42,43,44")
os.environ.setdefault("VARIANTES", "multi")
os.environ.setdefault("SALIDA", "confirmacion_v2.csv")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metodo_v2                                                    # noqa: E402,F401  (añade final/ al path)
import metodo                                                       # noqa: E402
os.environ.setdefault("DATASETS", ",".join(metodo.DATASETS))
import runpy                                                        # noqa: E402

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cribado_v2.py"), run_name="__main__")
