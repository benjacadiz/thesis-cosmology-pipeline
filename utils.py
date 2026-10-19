import numpy as np


IDS = {"flat_LCDM" : 0,
       "UG_barotropic" : 1,
       "UG_CSL" : 2,
       "UG_CSL_baryon": 3,
       "cos_cosh" : 4,
       "LCDM_baryon" : 5}
_warmup_registry = []

def register_warmup(func):
    _warmup_registry.append(func)
    return func
def run_warmup():
    for func in _warmup_registry:
        func()

def _get_param(theta, param, name, default=None):
    name_lower = name.lower()
    params_lower = [p.lower() for p in param]
    try:
        return theta[params_lower.index(name_lower)]
    except ValueError:
        if default is not None:
            return default
        raise KeyError(f"Parámetro '{name}' no encontrado en la lista de parámetros y no tiene default: {param}"
                         f"Parámetros disponibles: {param}")
    

def _get_model_id(model : str) -> int:  
    if model not in IDS:
        raise ValueError(f"Modelo no reconocido: {model}"
                         f"Modelos disponibles: {list(IDS.keys())}")
    return np.int64(IDS[model])
