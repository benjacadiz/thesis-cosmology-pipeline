from numba import njit
import numpy as np
from observables import mu_theoretical, Hz_theoretical,Ddt_theoretical, Dd_theoretical, Da_theoretical
from cosmology import dv_rd, dm_rd, dh_rd
from utils import _get_param, register_warmup

G = 4.3009e-9 # Mpc M_sun^-1 (km/s)^2
c = 299792.458 # km/s
alpha_bhs = 3.0 * np.sqrt(3) # Factor de Schwarzschild para el radio de la sombra
uas_to_rad = np.pi/(180.0*3600.0*1e6) # Conversión de microarcosegundos a radianes
bhs_factor = alpha_bhs * G / c**2

PRIOR_RANGES = {
    "h0" : (40.0, 100.0), # Constante de Hubble en km/s/Mpc
    "omega_m" : (0.0, 1.0), #Densidad de materia total
    "m" : (-21.0, -17.0), # Magnitud absoluta de supernova tipo Ia
    "x" : (-1.0, 1.0), # Entra en UG_barotropic
    "omega_xi" : (0.0, 1.0), #Entra en UG_CSL sin bariones
    "wb" : (0.0, 0.1), # El que entra en la fórmula de rd de DESI, no la densidad de bariones total,
    "omega_b0" : (0.0, 1.0), # Densidad de bariones
    "omega_dm0" : (0.0, 1.0), # Densidad de materia oscura
    "xi_tilde" : (0.0, 3.0), # Entra en UG_CSL_baryon
    "w_a" : (-2, 5), # Parámetro de EoS de energía oscura en cos_cosh
    "w_b" : (-2, 2), # Parámetro de EoS de energía oscura en cos_cosh
    "w_c" : (-2, 4)  # Parámetro de EoS de energía oscura en cos_cosh
}

def log_prior(theta : np.ndarray, params : list[str]):
    # Prior planos dentro de los rangos estándar. Considera el modelo para definir parámetros
    params = [p.lower() for p in params]
    for value, name in zip(theta, params):
            try:
                lower, upper = PRIOR_RANGES[name]
            except KeyError:
                raise KeyError(f"Parámetro desconocido en prior: {name}"
                               "Parámetros disponbles: " + ", ".join(PRIOR_RANGES.keys()))
            if not (lower <= value <= upper):
                return -np.inf
            
    lp = 0.0
    if "omega_dm0" in params and "omega_b0" in params:
        omega_dm0 = _get_param(theta, params, "omega_dm0")
        omega_b0 = _get_param(theta, params, "omega_b0")
        if omega_dm0 + omega_b0 >= 1.0:
            return -np.inf
    if "omega_xi" in params:
        omega_xi = _get_param(theta, params, "omega_xi")
        omega_m = _get_param(theta, params, "omega_m")
        if omega_m + 2.0*np.sqrt(omega_xi) >= 1.0:
            return -np.inf

    # Prior gaussiano de wb de Big Bang Nucleosynthesis (BBN)
    if "omega_b0" in params:
        omega_b0 = _get_param(theta, params, "omega_b0")
        h = _get_param(theta, params, "h0") / 100.0
        wb = omega_b0 * h**2
        lp += -0.5 * ((wb - 0.02218) / 0.00055)**2
    else:
        if not ("wb" in params):
            raise ValueError("No se encontró el parámetro 'wb' ni 'omega_b0' en los parámetros del modelo")
        
        wb = _get_param(theta, params, "wb")
        lp += -0.5 * ((wb - 0.02218) / 0.00055)**2
    
    return lp

def log_likelihood_ohd(theta : np.ndarray, params : list[str], data : dict, model : str = "flat_LCDM"):
    # Log-verosimilitud OHD, gaussiana simple. Construye residuos,
    # y retorna -chi2 / 2

    Hz_Th = Hz_theoretical(data["z"], theta, params, model)
    chi2 = _chi2_ohd_numba(data["Hz"], Hz_Th, data["sigmaH"])
    return -0.5 * chi2
        
def log_likelihood_sn(theta : np.ndarray, params : list[str], data : dict, model : str = "flat_LCDM"):
    
    M = _get_param(theta, params, "M")
    mu_all = mu_theoretical(data["z"], theta, params, model)
    mu_th = _mu_theoretical_full(mu_all, data["CEPH_DIST"], data["IS_CALIBRATOR"], M)

    chi2 = _chi2_pantheon(mu_th, data["m_b_corr"], data["cov_inv"])
    return -0.5 * chi2
                

def log_likelihood_h0licow(theta: np.ndarray, params: list[str], data: dict, model: str = "flat_LCDM"):
    
    ll = 0.0
    for lens_name, lens in data.items():
        zl = lens["zl"]
        zs = lens["zs"]

        if lens["type"] == "sln":
            Ddt_th = Ddt_theoretical(zl, zs, theta, params, model)
            ll += _log_likelihood_sln(Ddt_th, **lens["ddt_params"])

            if lens["dd_params"] is not None:
                Dd_th = Dd_theoretical(zl, theta, params, model)
                ll += _log_likelihood_sln(Dd_th, **lens["dd_params"])

        elif lens["type"] == "hist1d":
            Ddt_th = Ddt_theoretical(zl, zs, theta, params, model)
            ll += _log_likelihood_hist1d(Ddt_th, lens["hist1d"])

        elif lens["type"] == "hist2d":
            Ddt_th = Ddt_theoretical(zl, zs, theta, params, model)
            Dd_th = Dd_theoretical(zl, theta, params, model)
            ll += _log_likelihood_hist2d(Dd_th, Ddt_th, lens["hist2d"])
            
        else:
            raise ValueError(f"Tipo de likelihood desconocido para lente {lens_name}: {lens['type']}")
        
        if not np.isfinite(ll):
            return -np.inf
    return ll

def log_likelihood_desi(theta: np.ndarray, params: list[str], data: dict, model: str = "flat_LCDM"):
    H0 = _get_param(theta, params, "h0")
    h = H0 / 100.0

    if model in ["LCDM_baryon", "UG_CSL_baryon"]:
        Omega_b0 = _get_param(theta, params, "omega_b0")
        Omega_dm0 = _get_param(theta, params, "omega_dm0")
        omega_m_desi = (Omega_b0 + Omega_dm0) * h**2
        wb = Omega_b0 * h**2
    else:
        Omega_m = _get_param(theta, params, "omega_m")
        omega_m_desi = Omega_m * h**2
        wb = _get_param(theta, params, "wb", default = 0.02236)

    rd = _rd_desi(omega_m_desi, wb)

    th_list = []
    for tracer in data["tracers"]:
        zeff = tracer["zeff"]
        for qty in tracer["quantities"]:
            if qty == "DV":
                th_list.append(dv_rd(zeff, theta, params, rd, model))
            elif qty == "DM":
                th_list.append(dm_rd(zeff, theta, params, rd, model))
            elif qty == "DH":
                th_list.append(dh_rd(zeff, theta, params, rd, model))
            else:
                raise ValueError(f"Cantidad desconocida para DESI: {qty}")

    th_vec = np.array(th_list)
    delta = data["obs_vec"] - th_vec

    if not np.all(np.isfinite(delta)):
        return -np.inf
    return -0.5 * float(delta @ (data["cov_inv"] @ delta))

def log_likelihood_bhs(theta: np.ndarray, params: list[str], data: dict, model: str = "flat_LCDM"):

    ll = 0.0
    for k in range(len(data["z"])):
        z = data["z"][k]
        R_obs = data["R"][k]
        dR = data["dR"][k]
        M_obs = data["M"][k]
        dM = data["dM"][k]

        Da_obs, sigma_Da = _Da_from_M_theta(M_obs, R_obs, dM, dR)
        Da_th = Da_theoretical(z, theta, params, model)

        if not np.isfinite(Da_obs) or sigma_Da <= 0.0:
            continue
        res = (Da_obs - Da_th) / sigma_Da
        ll += -0.5 * res * res

        if not np.isfinite(ll):
            return -np.inf
    return ll


def log_likelihood_combined(theta : np.ndarray, params : list[str], datasets : dict, model : str = "flat_LCDM"):
    
    ll = 0.0
    for name, data in datasets.items():
        key = name.lower()
        if key not in LIKELIHOODS_REGISTRY:
            raise ValueError(f"Dataset desconocido: {key}"
                             f"Datasets disponibles: {list(LIKELIHOODS_REGISTRY.keys())}")
        ll += LIKELIHOODS_REGISTRY[key](theta, params, data, model)
        if not np.isfinite(ll):
            return -np.inf
    return ll

def log_posterior(theta : np.ndarray, params : list[str], datasets : dict, model : str = "flat_LCDM"):
    
    lp = log_prior(theta, params)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood_combined(theta, params, datasets, model)
    if not np.isfinite(ll):
        return -np.inf
    
    return lp + ll

@njit(fastmath = True)
def _chi2_ohd_numba(H_obs : np.ndarray, H_th : np.ndarray, sigma : np.ndarray):
    chi2 = 0.0
    n = H_obs.size
    for k in range(n):
        if sigma[k] <= 0.0:
            continue
        res = (H_obs[k] - H_th[k]) / sigma[k]
        chi2 += res * res
    return chi2

def _chi2_pantheon(mu_th : np.ndarray, m_b_corr : np.ndarray, inv_cov : np.ndarray):
    delta = m_b_corr - mu_th
    chi2 = delta @ (inv_cov @ delta)
    return chi2

@njit(fastmath=True)
def _mu_theoretical_full(mu_sn : np.ndarray, mu_ceph : np.ndarray, is_calib : np.ndarray, M : float):

    n = mu_sn.size
    mu_th = np.empty(n)
    for k in range(n):
        if is_calib[k]:
            mu_th[k] = mu_ceph[k] + M
        else:
            mu_th[k] = mu_sn[k] + M
    return mu_th

@njit(fastmath=True)
def _log_likelihood_sln(D_th: float, lambda_: float, mu: float, sigma: float):
    x = D_th - lambda_
    if x <= 0.0:
        return -np.inf
    return -np.log(sigma*x) - (np.log(x) - mu)**2 / (2.0 * sigma**2)

def _log_likelihood_hist2d(Dd_th: float, Ddt_th: float, hist2d_data):
    dd_min, dd_max = hist2d_data["dd_range"]
    ddt_min, ddt_max = hist2d_data["ddt_range"]

    if not (dd_min <= Dd_th <= dd_max) or not (ddt_min <= Ddt_th <= ddt_max):
        return -np.inf
    
    log_density = float(hist2d_data["interpolator"]([[Dd_th, Ddt_th]])[0])
    if not np.isfinite(log_density):
        return -np.inf
    return log_density

def _log_likelihood_hist1d(Ddt_th: float, hist1d_data):
    ddt_min, ddt_max = hist1d_data["ddt_range"]
    if not (ddt_min <= Ddt_th <= ddt_max):
        return -np.inf
    
    log_density = float(hist1d_data["interpolator"]([Ddt_th])[0])
    if not np.isfinite(log_density):
        return -np.inf
    return log_density

@njit(fastmath=True)
def _Da_from_M_theta(M, R, dM, dR):
    theta_rad = R * uas_to_rad
    Da_obs = bhs_factor * M / theta_rad
    sigma_Da = Da_obs * np.sqrt((dM/M)**2 + (dR/R)**2)
    return Da_obs, sigma_Da

def _rd_desi(omega_m_desi, wb, N_eff = 3.044):
    return 147.05 * (wb / 0.02236)**(-0.13) * (omega_m_desi / 0.1432)**(-0.23) * (N_eff / 3.04)**(-0.1)

#def aic(): pass
#def bic(): pass
#def evidence(): pass


LIKELIHOODS_REGISTRY = {
    "ohd" : log_likelihood_ohd,
    "cc" : log_likelihood_ohd,
    "pantheon" : log_likelihood_sn,
    "pantheon+" : log_likelihood_sn,
    "h0licow" : log_likelihood_h0licow,
    "desi" : log_likelihood_desi,
    "bhs" : log_likelihood_bhs
}

@register_warmup
def warmup():

    n = 50
    Hz_th = np.linspace(70.0, 200.0, n)
    Hz_obs = Hz_th + np.random.randn(n)* 3.0
    sigma = np.full(n, 3.0)
    _chi2_ohd_numba(Hz_obs, Hz_th, sigma)

    mu_th = np.linspace(31.0, 45.0, n)
    is_calib = np.zeros(n, dtype=np.bool_)
    is_calib[:5] = True
    ceph_dist = np.where(is_calib, 32.0, 0.0)

    _mu_theoretical_full(mu_th, ceph_dist, is_calib, M =-19.3)
    print("likelihoods.py: Numba kernels warmed up")

if __name__ == "__main__":
    warmup()