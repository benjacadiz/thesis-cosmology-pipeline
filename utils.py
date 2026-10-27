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

def _build_param_vector(model_id, theta, params):
    if model_id == 0: # flat_LCDM
        return np.array([_get_param(theta, params, "H0"),
                         _get_param(theta, params, "Omega_m")])
    elif model_id == 1:  # UG_barotropic
        return np.array([_get_param(theta, params, "H0"),
                         _get_param(theta, params, "Omega_m"),
                         _get_param(theta, params, "x")])
    elif model_id == 2:  # UG_CSL
        return np.array([_get_param(theta, params, "H0"),
                         _get_param(theta, params, "Omega_m"),
                         _get_param(theta, params, "omega_xi")])
    elif model_id == 3:  # UG_CSL_baryon
        return np.array([_get_param(theta, params, "H0"),
                         _get_param(theta, params, "Omega_b0"),
                         _get_param(theta, params, "Omega_dm0"),
                         _get_param(theta, params, "xi_tilde")])
    elif model_id == 4:  # cos_cosh
        return np.array([_get_param(theta, params, "H0"),
                         _get_param(theta, params, "Omega_m"),
                         _get_param(theta, params, "w_a"),
                         _get_param(theta, params, "w_b"),
                         _get_param(theta, params, "w_c")])
    elif model_id == 5:  # LCDM_baryon
        return np.array([_get_param(theta, params, "H0"),
                         _get_param(theta, params, "Omega_dm0"),
                         _get_param(theta, params, "Omega_b0")])
    else:
        raise ValueError(f"Modelo no reconocido con ID {model_id}")    

def _get_model_id(model : str) -> int:  
    if model not in IDS:
        raise ValueError(f"Modelo no reconocido: {model}"
                         f"Modelos disponibles: {list(IDS.keys())}")
    return np.int64(IDS[model])
