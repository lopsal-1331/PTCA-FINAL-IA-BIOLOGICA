"""
compat.py — Compatibilidad de metric-learn y gplearn con scikit-learn >= 1.6.

Ambas librerías llaman a check_X_y/check_array con el argumento `force_all_finite`, que scikit-learn renombró a
`ensure_all_finite`. Este módulo traduce el argumento antiguo al nuevo. Hay que importarlo ANTES que esas librerías:
    import compat  # noqa: F401
"""
import functools
import sklearn.utils as U
import sklearn.utils.validation as V


def _traducir(f):
    @functools.wraps(f)
    def g(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return f(*args, **kwargs)
    return g


def aplicar():
    """Parchea scikit-learn y, por si ya se importaron, los módulos de metric_learn y gplearn (que guardan su propia
    referencia a check_X_y/check_array). Es idempotente; se llama al importar y al empezar cada tarea en paralelo."""
    import sys
    for nombre in ["check_X_y", "check_array"]:
        f = getattr(V, nombre)
        if not getattr(f, "_compat", False):
            f = _traducir(f)
            f._compat = True
            setattr(V, nombre, f)
        setattr(U, nombre, f)
        for mod in list(sys.modules.values()):
            if mod is not None and getattr(mod, "__name__", "").split(".")[0] in ("metric_learn", "gplearn") \
                    and hasattr(mod, nombre):
                setattr(mod, nombre, f)


aplicar()
