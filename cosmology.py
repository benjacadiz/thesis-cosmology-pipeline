import numpy as np
from numbalsoda import lsoda_sig, lsoda
from numba import njit, prange, cfunc
from typing import Any
from utils import _get_param, _get_model_id, register_warmup, IDS
from scipy.interpolate import interp1d
from scipy.optimize import brentq
c = 299792.458 # Speed of light in km/s
G = 4.3009e-9
S_PER_GYR =  3.15576e16
KM_PER_MPC = 3.08567e19
_csl_cache = {}
_csl_baryon_cache = {}
_csl_baryon_cache_z = {}
"""

"""

def E(z : float, theta : np.ndarray, params : list, model : str = "flat_LCDM") -> float:
    """
    Function that returns the dimensionless Hubble parameter E(z) = H(z) / H0
    for a given redshift z and cosmological parameters theta. The model by default
    is the flat Lambda CDM model for late universe.

    Parameters
    ----------
    z : float
        Redshift at which to evaluate E(z).
    theta : np.ndarray
        Array of cosmological parameters.
    model : str, optional
        Cosmological model to use. The default is 'flat_LCDM'. 
        Other models implemented are 'UG_barotropic' and 'UG_CSL'.

    Returns
    -------
    float
        The dimensionless Hubble parameter E(z) for the given redshift and cosmological parameters

    """
    if model == "flat_LCDM":
        Omega_m = _get_param(theta, params, "Omega_m")
        return _E_flat_lcdm(z, Omega_m)
    elif model == "UG_barotropic":
        Omega_m = _get_param(theta, params, "Omega_m")
        x = _get_param(theta, params, "x")
        return _E_barotropic_model(z, Omega_m, x)
    elif model == "UG_CSL":
        raise ValueError("E(z) no disponible directamente para UG_CSL." \
                        "Usar H(z) o comoving_distance para obtener E(z) indirectamente.")
    elif model == "UG_CSL_baryon":
        raise ValueError("E(z) no disponible directamente para UG_CSL_baryon." \
                        "Usar H(z) o comoving_distance para obtener E(z) indirectamente.")
    elif model == "cos_cosh":
        Omega_m = _get_param(theta, params, "Omega_m")
        w_a = _get_param(theta, params, "w_a")
        w_b = _get_param(theta, params, "w_b")
        w_c = _get_param(theta, params, "w_c")
        return _E_coscosh_model(z, Omega_m, w_a, w_b, w_c)
    else:
        raise ValueError(f"Modelo no reconocido: {model}"
                         f"Modelos disponibles: {list(IDS.keys())}")

def H(z: float, theta: np.ndarray, params: list, model = "flat_LCDM") -> float:
    """
    Function that returns the Hubble parameter H(z) for a given redshift z and cosmological parameters theta.
    The model by default is the flat Lambda CDM model for late universe.
    
    Parameters
    ----------
    z : float
        Redshift at which to evaluate H(z).
    theta : np.ndarray
        Array of cosmological parameters.
    model : str, optional
        Cosmological model to use. The default is "flat_LCDM".
        Other models implemented are 'UG_barotropic' and 'UG_CSL'.
        For CSL, the Hubble parameter is calculated numerically and cached
        for efficency.
    
    Returns
    -------
    float
        The Hubble parameter H(z) for the given redshift and cosmological parameters
    """

    H0 = _get_param(theta, params, "H0")
    if model == "UG_CSL":
        omega_xi = _get_param(theta, params, "omega_xi")
        omega_m = _get_param(theta, params, "omega_m")
        H_interp, _ = _get_csl_interpolators(H0, omega_m, omega_xi)
        return float(H_interp(np.atleast_1d(np.asarray(z, dtype=np.float64)))[0])
    elif model == "UG_CSL_baryon":
        Omega_b0 = _get_param(theta, params, "Omega_b0")
        Omega_dm0 = _get_param(theta, params, "Omega_dm0")
        xi_tilde = _get_param(theta, params, "xi_tilde")
        H_interp, _ = _get_csl_baryon_interpolators_z(H0, Omega_b0, Omega_dm0, xi_tilde)
        return float(H_interp(np.atleast_1d(np.asarray(z, dtype=np.float64)))[0])

    return H0 * E(z, theta, params, model)

def comoving_distance(z: float, theta: np.ndarray, params: list, model: str = "flat_LCDM") -> float:
    """
    Function that returns the comoving distance D_c(z) for a given redshift z and cosmological parameters theta.
    The model by default is the flat Lambda CDM model for late universe.
    The comoving distance is:
        D_c(z) = c / H0 * ∫_0^z dz' / E(z')
    This is calculated using the trapezoidal rule with n_points points.

    Parameters
    ----------
    z : float
        Redshift at which to evaluate the comoving distance.
    theta : np.ndarray
        Array of cosmological parameters.
    model : str, optional
        Cosmological model to use. The default is "flat_LCDM".
        Other models implemented are 'UG_barotropic' and 'UG_CSL'.
        For CSL, the comoving distance is calculated numerically and cached
        for efficency.
    
    
    Returns
    -------
    float
        The comoving distance D_c(z) for the given redshift and cosmological parameters
    """
    H0 = _get_param(theta, params, "H0")
    
    x = 0.0
    w_a = 0.0
    w_b = 0.0
    w_c = 0.0
    model_id = np.int64(_get_model_id(model))
    if model_id == 0:
        Omega_m = _get_param(theta, params, "Omega_m")
    if model_id == 1: # barotropic UG
        Omega_m = _get_param(theta, params, "Omega_m")
        x = _get_param(theta, params, "x")
    elif model_id == 2: 
        omega_xi = _get_param(theta, params, "omega_xi")
        Omega_m = _get_param(theta, params, "Omega_m")
        _, dc_interp = _get_csl_interpolators(H0, Omega_m, omega_xi)
        return float(dc_interp(np.atleast_1d(np.asarray(z, dtype=np.float64)))[0])
    elif model_id == 3:
        Omega_b0 = _get_param(theta, params, "Omega_b0")
        Omega_dm0 = _get_param(theta, params, "Omega_dm0")
        xi_tilde = _get_param(theta, params, "xi_tilde")
        _, dc_interp = _get_csl_baryon_interpolators_z(H0, Omega_b0, Omega_dm0, xi_tilde)
        return float(dc_interp(np.atleast_1d(np.asarray(z, dtype=np.float64)))[0])
    elif model_id == 4: #cos_cosh
        w_a = _get_param(theta, params, "w_a")
        w_b = _get_param(theta, params, "w_b")
        w_c = _get_param(theta, params, "w_c")
        Omega_m = _get_param(theta, params, "Omega_m")
    
    return _comoving_distance_single(float(z), H0, Omega_m, x, w_a, w_b, w_c, model_id)

def comoving_distance_array(z_array : np.ndarray[Any, np.dtype[np.float64]], theta : np.ndarray, params: list, model : str = "flat_LCDM") -> np.ndarray[Any, np.dtype[np.float64]]:
    H0 = _get_param(theta, params, "H0")

    x = 0.0
    w_a = 0.0
    w_b = 0.0
    w_c = 0.0
    model_id = np.int64(_get_model_id(model))
    if model_id == 0:
        Omega_m = _get_param(theta, params, "Omega_m")
    if model_id == 1: # barotropic UG
        Omega_m = _get_param(theta, params, "Omega_m")
        x = _get_param(theta, params, "x")
    elif model_id == 2:
        omega_xi = _get_param(theta, params, "omega_xi")
        Omega_m = _get_param(theta, params, "Omega_m")
        _, dc_interp = _get_csl_interpolators(H0, Omega_m, omega_xi)
        return dc_interp(z_array.astype(np.float64))
    elif model_id == 3:
        Omega_b0 = _get_param(theta, params, "Omega_b0")
        Omega_dm0 = _get_param(theta, params, "Omega_dm0")
        xi_tilde = _get_param(theta, params, "xi_tilde")
        _, dc_interp = _get_csl_baryon_interpolators_z(H0, Omega_b0, Omega_dm0, xi_tilde)
        return dc_interp(z_array.astype(np.float64))
    elif model_id == 4:
        w_a = _get_param(theta, params, "w_a")
        w_b = _get_param(theta, params, "w_b")
        w_c = _get_param(theta, params, "w_c")
        Omega_m = _get_param(theta, params, "Omega_m")

    sorted_idx = np.argsort(z_array)
    z_sorted = z_array[sorted_idx].astype(np.float64)
    d_sorted = _comoving_distance_array(z_sorted, H0, Omega_m, x, w_a, w_b, w_c, model_id)

    result = np.empty_like(z_array)
    result[sorted_idx] = d_sorted
    return result


def luminosity_distance(z_array : np.ndarray[Any, np.dtype[np.float64]], theta : np.ndarray, params: list, model : str = "flat_LCDM") -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Function that calculates the luminosity distance D_l(z) for a given redshift z and
    cosmological parameters theta. The model by default is the flat Lambda CDM model for
    late universe.
    The luminosity distance is related to the comoving distance by:
        D_l(z) = (1 + z) * D_c(z)
    
    Parameters
    ----------
    z : np.ndarray
        Array of redshifts at which to evaluate the luminosity distance.
    theta : np.ndarray
        Array of cosmological parameters.
    model : str, optional
        Cosmological model to use. The default is "flat_LCDM".
        Other models implemented are 'UG_barotropic' and 'UG_CSL'.

    Returns
    -------
    np.ndarray
        The luminosity distance D_l(z) for the list of redshifts and cosmological parameters.

    """
    dc = comoving_distance_array(z_array, theta, params, model)
    return (1.0 + z_array) * dc

def angular_diameter_distance(z: float, theta : np.ndarray, params: list, model : str="flat_LCDM") -> float:
    """
    Function that calculates the angular diameter distancde D_A(z) for a given redshift z and cosmological
    parameters theta. The angular diameter distance is calculated as
        D_A(z) = D_c(z) / (1 + z)
        
    where D_c(z) is the comoving distance.
    The model by default is the flat LambdaCDM model for late universe.
    
    Parameters
    ----------
    z: float
        Redshift at which to evaluate the angular diameter distance.
    theta : np.ndarray
        Array of cosmological parameters.
    model : str, optional
        Cosmological model to use. The default is "flat_LCDM".
        Other models implemented are 'UG_barotropic' and 'UG_CSL'.

    Returns
    -------
    float
        The angular diameter distance D_A(z) for the given redshift and cosmological parameters.
    """
    dc = comoving_distance(z, theta, params, model)
    return dc / (1.0 + z)

def ad_between_distances(z1: float, z2 : float, theta : np.ndarray, params: list, model : str = "flat_LCDM") -> float:

    if z2 <= z1:
        raise ValueError(f"z2 debe ser mayor que z1. Recibido: z1={z1}, z2={z2}")
    
    H0 = _get_param(theta, params, "H0")
    
    x = 0.0
    w_a = 0.0
    w_b = 0.0
    w_c = 0.0
    model_id = np.int64(_get_model_id(model))
    if model_id == 0:
        Omega_m = _get_param(theta, params, "Omega_m")
    if model_id == 1: # barotropic UG
        Omega_m = _get_param(theta, params, "Omega_m")
        x = _get_param(theta, params, "x")
    elif model_id == 2: # CSL UG
        omega_xi = _get_param(theta, params, "omega_xi")
        Omega_m = _get_param(theta, params, "Omega_m")
        _, dc_interp = _get_csl_interpolators(H0, Omega_m, omega_xi)
        dc_z1 = float(dc_interp(np.atleast_1d(np.asarray(z1, dtype=np.float64)))[0])
        dc_z2 = float(dc_interp(np.atleast_1d(np.asarray(z2, dtype=np.float64)))[0])
        return (dc_z2 - dc_z1) / (1.0 + z2)
    elif model_id == 3: # CSL baryon UG
        Omega_b0 = _get_param(theta, params, "Omega_b0")
        Omega_dm0 = _get_param(theta, params, "Omega_dm0")
        xi_tilde = _get_param(theta, params, "xi_tilde")
        _, dc_interp = _get_csl_baryon_interpolators_z(H0, Omega_b0, Omega_dm0, xi_tilde)
        dc_z1 = float(dc_interp(np.atleast_1d(np.asarray(z1, dtype=np.float64)))[0])
        dc_z2 = float(dc_interp(np.atleast_1d(np.asarray(z2, dtype=np.float64)))[0])
        return (dc_z2 - dc_z1) / (1.0 + z2)
    elif model_id == 4:
        Omega_m = _get_param(theta, params, "Omega_m")
        w_a = _get_param(theta, params, "w_a")
        w_b = _get_param(theta, params, "w_b")
        w_c = _get_param(theta, params, "w_c")

    return _angular_diameter_distance_between(float(z1), float(z2), H0, Omega_m, x, w_a, w_b, w_c, model_id)

def dv_rd(z: float, theta: np.ndarray, params: list, rd: float, model: str = "flat_LCDM") -> float:
    """
    Function that calculates the volume-averaged distance D_V(z) for a given redshift z and cosmological parameters
    theta. The volume-averaged distance is defined as:
        D_V(z) = [c * z / H(z) * D_M(z)^2]^(1/3)
    where D_M(z) is the comoving distance. The function returns D_V(z) / r_d, where r_d is the sound horizon at the given redshift.
    
    Parameters
    ----------
    z: float
        Redshift at which to evaluate the volume-averaged distance.
    theta : np.ndarray
        Array of cosmological parameters.
    params : list
        List of parameter names.
    rd : float
        Sound horizon at the given redshift.
    model : str, optional
        Cosmological model to use. The default is "flat_LCDM".
        Other models implemented are 'UG_barotropic' and 'UG_CSL'.

    Returns
    -------
    float
        The volume-averaged distance D_V(z) / r_d for the given redshift and cosmological parameters.
    """
    Hz = H(z, theta, params, model)
    DM = comoving_distance(z, theta, params, model)
    DV = (c*z/Hz * DM**2) ** (1.0/3.0)
    return DV / rd

def dm_rd(z: float, theta: np.ndarray, params: list, rd: float, model: str = "flat_LCDM") -> float:
    """
    Function that calculates the comoving distance D_M(z) for a given redshift z and cosmological parameters.
    The function returns D_M(z) / r_d, where r_d is the sound horizon at the given redshift.
    
    Parameters
    ----------
    z: float
        Redshift at which to evaluate the comoving distance.
    theta : np.ndarray
        Array of cosmological parameters.
    params : list
        List of parameter names.
    rd : float
        Sound horizon at the given redshift.
    model : str, optional
        Cosmological model to use. The default is "flat_LCDM".
        Other models implemented are 'UG_barotropic' and 'UG_CSL'.

    Returns
    -------
    float
        The comoving distance D_M(z) / r_d for the given redshift and cosmological parameters.
    """
    DM = comoving_distance(z, theta, params, model)
    return DM / rd

def dh_rd(z: float, theta: np.ndarray, params: list, rd: float, model: str = "flat_LCDM") -> float:
    Hz = H(z, theta, params, model)
    return c / (Hz * rd)

def H_array(z_array: np.ndarray, theta: np.ndarray, params: list, model: str = "flat_LCDM") -> np.ndarray:
    H0 = _get_param(theta, params, "H0")
    model_id = np.int64(_get_model_id(model))
    x = 0.0
    w_a = 0.0
    w_b = 0.0
    w_c = 0.0
    if model_id == 0:
        omega_m = _get_param(theta, params, "Omega_m")

    if model_id == 1:
        omega_m = _get_param(theta, params, "Omega_m")
        x = _get_param(theta, params, "x")
    elif model_id == 2:
        omega_xi = _get_param(theta, params, "omega_xi")
        omega_m = _get_param(theta, params, "Omega_m")
        H_interp, _ = _get_csl_interpolators(H0, omega_m, omega_xi)
        return H_interp(z_array.astype(np.float64))
    elif model_id == 3:
        Omega_b0 = _get_param(theta, params, "Omega_b0")
        Omega_dm0 = _get_param(theta, params, "Omega_dm0")
        xi_tilde = _get_param(theta, params, "xi_tilde")
        H_interp, _ = _get_csl_baryon_interpolators_z(H0, Omega_b0, Omega_dm0, xi_tilde)
        return H_interp(z_array.astype(np.float64))
    elif model_id == 4:
            omega_m = _get_param(theta, params, "Omega_m")
            w_a = _get_param(theta, params, "w_a")
            w_b = _get_param(theta, params, "w_b")
            w_c = _get_param(theta, params, "w_c")
    return _H_array_numba(z_array, H0, omega_m, x, w_a, w_b, w_c, model_id)


@njit(fastmath=True)
def _H_array_numba(z_array, H0, omega_m, x, w_a, w_b, w_c, model_id):
    n_z = z_array.size
    H_array = np.empty(n_z)
    for k in range(n_z):
        zk = z_array[k]
        Ez = _E_dispatch(zk, omega_m, x, w_a, w_b, w_c, model_id)
        H_array[k] = H0 * Ez
    return H_array

@njit(fastmath = True)
def _E_flat_lcdm(z : float, omega_m : float) -> float:
    return (omega_m * (1.0 + z)**3 + (1.0 - omega_m)) ** 0.5

@njit(fastmath=True)
def _E_barotropic_model(z : float, omega_m : float,  x : float) -> float:
    omega_lambda = 1.0 - omega_m - x*omega_m
    return (omega_m * (1.0 + x) * (1.0 + z)**(3/(1.0+x)) + omega_lambda) ** 0.5

@njit(fastmath=True)
def _E_coscosh_model(z: float, omega_m: float, w_a: float, w_b: float, w_c: float) -> float:
    a = 1.0 / (1.0 + z)
    rho_de = np.exp(-3 * ((1-w_c) * np.log(a) - w_b * (np.sinh(a) - np.sinh(1)) + w_a * (np.sin(a) - np.sin(1))))
    return (omega_m * (1.0 + z)**3 + (1.0 - omega_m) * rho_de) ** 0.5
@njit(fastmath=True)
def _E_dispatch(z, omega_m, x, w_a = 0, w_b = 0, w_c = 0, model_id = 0):
    if model_id == 0:
        return _E_flat_lcdm(z, omega_m)
    elif model_id == 1:
        return _E_barotropic_model(z, omega_m, x)
#    elif model_id == 2:
#        return _E_csl(z, omega_m, x)
    elif model_id == 4:
        return _E_coscosh_model(z, omega_m, w_a, w_b, w_c)
    else:
        return -1.0
    
@njit(fastmath=True)
def _comoving_distance_single(z : float, H0 : float, omega_m : float, x : float, w_a, w_b, w_c, model_id : int, n : int = 300) -> float:
    if z <= 0.0:
        return 0.0

    dz = z/n
    integral = 0.0
    for k in range(n+1):
        zp = k * dz
        w = 0.5 if (k==0 or k==n) else 1.0
        Ez = _E_dispatch(zp, omega_m, x, w_a, w_b, w_c, model_id)
        integral += w/Ez
    
    return (c/ H0) * dz * integral

@njit(parallel = True, fastmath=True)
def _comoving_distance_array(z_array : np.ndarray[Any, np.dtype[np.float64]], H0 : float, omega_m : float, x : float, w_a, w_b, w_c, model_id : int, n : int = 300) -> np.ndarray[Any, np.dtype[np.float64]]:

    z_max = z_array[-1]
    dz = z_max / (n-1)

    integrand = np.empty(n)
    for k in prange(n):
        zp = k * dz
        Ez = _E_dispatch(zp, omega_m, x, w_a, w_b, w_c, model_id)
        integrand[k] = 1.0 / Ez
    d_grid = np.empty(n)
    d_grid[0] = 0.0
    for k in range(1, n):
        d_grid[k] = d_grid[k-1] + 0.5 * (integrand[k-1] + integrand[k]) * dz
    d_grid *= c / H0

    n_z = z_array.size
    result = np.empty(n_z)
    for k in range(n_z):
        zk = z_array[k]
        if zk <= 0.0:
            result[k] = 0.0
            continue

        idx = zk / dz
        i0 = int(idx)
        if idx >= n-1:
            result[k] = d_grid[n-1]
            continue
            
        frac = idx - i0
        result[k] = d_grid[i0] * (1.0 - frac) + d_grid[i0+1] * frac
    return result

@njit(fastmath=True)
def _angular_diameter_distance_between(z1 : float, z2 : float, H0 : float, omega_m : float, x : float, w_a, w_b, w_c, model_id : int, n : int = 300) -> float:
    
    dz = (z2-z1) / n
    integral = 0.0
    for k in range(n+1):
        zp = z1 + k * dz
        w = 0.5 if (k==0 or k==n) else 1.0
        Ez = _E_dispatch(zp, omega_m, x, w_a, w_b, w_c, model_id)
        integral += w/Ez
    dc_12 = (c/ H0) * dz * integral
    return dc_12 / (1.0 + z2)

@njit(fastmath=True)
def _csl_constants(H0, omega_m, omega_xi, omega_l):
    s_ox = np.sqrt(omega_xi)
    s_oxl = np.sqrt(omega_xi + omega_l)
    mu = 3/2  * s_oxl

    A = (1 - s_ox - s_oxl)/(1 - s_ox + s_oxl)
    B = (1-s_ox) / s_oxl

    return s_ox, s_oxl, mu, A, B

    
def _find_tau_max(z_max, H0, sqrt_ox, mu, B):
    def f(tt):
        theta = mu * tt
        inner = np.cosh(theta) + B * np.sinh(theta)
        if inner <= 0.0:
            return np.inf
        a = np.exp(sqrt_ox * tt) * inner**(2.0/3.0)
        if a <= 0.0 or not np.isfinite(a):
            return np.inf
        return 1.0/a - 1.0 - z_max
    
    # tau=0: f = -z_max < 0
    # tau<<0: a->0, f -> +inf
    return brentq(f, -5, 0.0)


@njit(fastmath=True)
def _build_csl_grid(tau_tilde_grid, H0, s_ox, s_oxl, mu, B):
    n = tau_tilde_grid.size
    a_grid = np.empty(n)
    H_grid = np.empty(n)
    for k in range(n):
        tt = tau_tilde_grid[k]
        arg_hyp = mu*tt
        tanh = np.tanh(arg_hyp)
        cosh = np.cosh(arg_hyp)
        sinh = np.sinh(arg_hyp)
        parenthesis = cosh + B * sinh
        a_grid[k] = np.exp(s_ox*tt) * parenthesis**(2/3) if parenthesis > 0.0 else 0.0
        num = 1 + (s_ox - s_ox**2 + s_oxl**2)/s_oxl * tanh
        denom = 1 + B * tanh
        H_grid[k] = H0 * num / denom

    return a_grid, H_grid
    

def _get_csl_interpolators(H0, omega_m, omega_xi, n=300):
    key = (np.float64(H0), np.float64(omega_m), np.float64(omega_xi))
    if key in _csl_cache:
        return _csl_cache[key]
    
    omega_l = 1.0 - omega_m - 2.0*np.sqrt(omega_xi)
    if omega_l <= 0.0:
        return None
    
    sqrt_ox, sqrt_oxl, mu, A, B = _csl_constants(H0, omega_m, omega_xi, omega_l)
    tt_max = _find_tau_max(2.5, H0, sqrt_ox, mu, B)  # z_max fijo conservador
    tt_grid = np.linspace(tt_max, 0.0, n)
    a_grid, H_grid = _build_csl_grid(tt_grid, H0, sqrt_ox, sqrt_oxl, mu, B)
    z_grid = (1.0/a_grid - 1.0)[::-1]
    H_grid_z = H_grid[::-1]
    
    dc_grid = np.zeros(len(z_grid))
    for k in range(1, len(z_grid)):
        dc_grid[k] = dc_grid[k-1] + 0.5*(c/H_grid_z[k-1] + c/H_grid_z[k])*(z_grid[k]-z_grid[k-1])
    
    H_interp = interp1d(z_grid, H_grid_z, kind="cubic",
                        bounds_error=False, fill_value="extrapolate")
    dc_interp = interp1d(z_grid, dc_grid, kind="cubic",
                         bounds_error=False, fill_value="extrapolate")
    
    _csl_cache.clear()  # solo guardamos el último
    _csl_cache[key] = (H_interp, dc_interp)
    return H_interp, dc_interp

# Implementación de CSL con bariones, integrando en términos del tiempo tau e integrando hacia atrás hasta llegar a z_max
#
#@cfunc(lsoda_sig)
#def _csl_baryon_rhs_backwards(t, u, du, p):
#    # u = [E, rho_b , rho_dm, a]
#    # p = [xi_tilde]
#    # Esta expresión es para la integración atrás en el tiempo
#    # Habiendo definido las derivadas en tau, estas van en -tau
#
#    E = u[0]
#    rho_b = u[1]
#    rho_dm = u[2]
#    a = u[3]
#
#    xi_tilde = p[0]
#
#    du[0] = 3/2* (rho_b + rho_dm)  # dE/dt
#    du[1] = -rho_b * (xi_tilde - 3*E) # d(rho_b)/dt
#    du[2] = 3*E*rho_dm  # d(rho_dm)/dt
#    du[3] = -E * a  # da/dt
#
#_csl_baryon_rhs_ptr = _csl_baryon_rhs_backwards.address
#
#@njit(fastmath=True)
#def _integrate_csl_baryon(tau_grid, Omega_b0, Omega_dm0, xi_tilde, n=300):
#    u0 = np.array([1.0, Omega_b0, Omega_dm0, 1.0])
#    p = np.array([xi_tilde])
#    usol, success = lsoda(_csl_baryon_rhs_ptr, u0, tau_grid, data=p, rtol=1e-8, atol=1e-10)
#    return usol, success
#
#def _find_tau_max_baryon(z_max, Omega_b0, Omega_dm0, xi_tilde, tau_init = 0.3, n_test = 200, max_extend=50):
#
#    tau_lo = tau_init
#    for _ in range(max_extend):
#        tau_grid = np.linspace(0.0, tau_lo, n_test)
#        usol, success = _integrate_csl_baryon(tau_grid, Omega_b0, Omega_dm0, xi_tilde, n_test)
#        if not success:
#            tau_lo *= 0.8
#            continue
#        a_max = usol[-1,3]
#        if a_max <= 0.0 or not np.isfinite(a_max):
#            tau_lo *= 0.8
#            continue
#        z_reached = 1.0/a_max - 1.0
#        if z_reached >= z_max:
#            return tau_lo
#        tau_lo *= 1.15
#    raise RuntimeError(f"No se alcanzó z_max = {z_max} en el último tau_lo = {tau_lo}. Último z alcanzado = {z_reached}")
#
#def _get_csl_baryon_interpolators(H0, Omega_b0, Omega_dm0, xi_tilde, n=300):
#
#    key = (np.float64(H0), np.float64(Omega_b0), np.float64(Omega_dm0), np.float64(xi_tilde))
#    if key in _csl_baryon_cache:
#        return _csl_baryon_cache[key]
#    if Omega_b0 + Omega_dm0 >= 1.0:
#        return None
#    
#    tau_max = _find_tau_max_baryon(2.5, Omega_b0, Omega_dm0, xi_tilde)
#    tau_grid = np.linspace(0.0, tau_max, n)
#    usol, success = _integrate_csl_baryon(tau_grid, Omega_b0, Omega_dm0, xi_tilde, n)
#    if not success:
#        return None
#    
#    E_grid = usol[:, 0]
#    a_grid = usol[:, 3]
#
#    if np.any(a_grid <= 0.0) or np.any(~np.isfinite(a_grid)):
#        return None
#    
#    z_grid = (1.0/a_grid - 1.0)
#    Hz_grid = H0 * E_grid
#
#    idx = np.argsort(z_grid)
#    z_grid = z_grid[idx]
#    Hz_grid = Hz_grid[idx]
#
#    dc_grid = np.zeros(len(z_grid))
#    for k in range(1, len(z_grid)):
#        dc_grid[k] = dc_grid[k-1] + 0.5*(c/Hz_grid[k-1] + c/Hz_grid[k])*(z_grid[k]-z_grid[k-1])
#
#    H_interp = interp1d(z_grid, Hz_grid, kind="cubic",bounds_error=False, fill_value="extrapolate")
#    dc_interp = interp1d(z_grid, dc_grid, kind="cubic",bounds_error=False, fill_value="extrapolate")
#
#    _csl_baryon_cache.clear()
#    _csl_baryon_cache[key] = (H_interp, dc_interp)
#    return H_interp, dc_interp

@cfunc(lsoda_sig)
def _csl_baryon_rhs_z(z, u, du, p):
    # u = [E, rho_b , rho_dm, a, Q]
    # p = [xi_tilde]
    # Integración en términos de z, sin inversión de signo dado que el cambio de variable ya invierte la dirección.

    E=u[0]
    rho_b = u[1]
    rho_dm = u[2]
    a = u[3]
#    Q = u[4]


    xi_tilde = p[0]

    one_plus_z = 1.0 + z

    du[0] = 3.0/2.0 * (rho_b + rho_dm) / (one_plus_z*E)  # dE/dz
    du[1] = rho_b * (3*E - xi_tilde) / (one_plus_z*E) # d(rho_b)/dz
    du[2] = 3 * rho_dm / (one_plus_z)  # d(rho_dm)/dz
    du[3] = -a / (one_plus_z)  # da/dz
#    du[4] = xi_tilde * rho_b / (one_plus_z*E)   dQ/dz

_csl_baryon_rhs_z_ptr = _csl_baryon_rhs_z.address

@njit(fastmath=True)
def _integrate_csl_baryon_z(z_grid, Omega_b0, Omega_dm0, xi_tilde, n=300):
    u0 = np.array([1.0, Omega_b0, Omega_dm0, 1.0])
    p = np.array([xi_tilde])
    usol, success = lsoda(_csl_baryon_rhs_z_ptr, u0, z_grid, data=p, rtol=1e-8, atol=1e-10)
    return usol, success

def _get_csl_baryon_interpolators_z(H0, Omega_b0, Omega_dm0, xi_tilde, z_max = 2.5, n=300):
    
    key = (np.float64(H0), np.float64(Omega_b0), np.float64(Omega_dm0), np.float64(xi_tilde))
    if key in _csl_baryon_cache_z:
        return _csl_baryon_cache_z[key]
    if Omega_b0 + Omega_dm0 >= 1.0:
        return None
    
    z_grid = np.linspace(0.0, z_max, n)
    usol, success = _integrate_csl_baryon_z(z_grid, Omega_b0, Omega_dm0, xi_tilde, n)
    if not success:
        return None
    E_grid = usol[:, 0]
    Hz_grid = H0 * E_grid

    if np.any(Hz_grid <= 0.0) or np.any(~np.isfinite(Hz_grid)):
        return None
    
    dc_grid = np.zeros(n)
    for k in range(1, n):
        dc_grid[k] = dc_grid[k-1] + 0.5*(c/Hz_grid[k-1] + c/Hz_grid[k])*(z_grid[k]-z_grid[k-1])

    H_interp = interp1d(z_grid, Hz_grid, kind="cubic",bounds_error=False, fill_value="extrapolate")
    dc_interp = interp1d(z_grid, dc_grid, kind="cubic",bounds_error=False, fill_value="extrapolate")

    _csl_baryon_cache_z.clear()
    _csl_baryon_cache_z[key] = (H_interp, dc_interp)

    return H_interp, dc_interp

def compute_t0(H0, Omega_m, omega_xi, z_max_bb = 1e8, tau_search=(-50.0, 0.0)):
    omega_l = 1.0 - Omega_m - 2.0*np.sqrt(omega_xi)
    if omega_l <= 0.0:
        return np.nan
    
    sqrt_ox, sqrt_oxl, mu, A, B = _csl_constants(H0, Omega_m, omega_xi, omega_l)
    def f(tt):
        theta = mu * tt
        parenthesis = np.cosh(theta) + B * np.sinh(theta)
        if parenthesis <= 0.0:
            return np.inf
        a = np.exp(sqrt_ox * tt) * parenthesis**(2.0/3.0)
        if a <= 0.0 or not np.isfinite(a):
            return np.inf
        z = 1.0/a - 1.0
        return z - z_max_bb
    
    try:
        tau_bb = brentq(f, tau_search[0], tau_search[1])
    except ValueError:
        return np.nan
    t0_Gyr = -tau_bb / H0 * 977.8
    return t0_Gyr

def compute_t0_baryon(H0, Omega_b0, Omega_dm0, xi_tilde, z_max_bb = 1e8, n=2000):
    if Omega_b0 + Omega_dm0 >= 1.0:
        return np.nan
    z_grid = np.concatenate([[0.0], np.geomspace(1e-3, z_max_bb, n-1)])
    usol, success = _integrate_csl_baryon_z(z_grid, Omega_b0, Omega_dm0, xi_tilde, n)
    if not success:
        return np.nan
    
    E_grid = usol[:, 0]
    a_grid = usol[:, 3]

    if np.any(E_grid <= 0.0) or np.any(~np.isfinite(E_grid)):
        return np.nan
    
    if a_grid[-1] > 1e-3:
        return np.nan # No se alcanzó z_max_bb, esperamos un a cercano a 0
    print(f"a_grid[-1] = {a_grid[-1]} reached at z = {1.0/a_grid[-1]-1.0}, expected z_max_bb = {z_max_bb}")
    integrand = 1.0 / ((1.0+z_grid) * E_grid)

    t0_s = np.trapz(integrand, z_grid) / H0 * KM_PER_MPC
    t0_Gyr = t0_s / S_PER_GYR
    return t0_Gyr

def q_z(z_array, theta, params, model="flat_LCDM", dz = 1e-4):
    z_array = np.asarray(z_array, dtype=np.float64)
    z_low = max(z_array.min() - 5*dz, 0.0)
    z_high = z_array.max() + 5*dz
    n_dense = int((z_high - z_low)/dz) + 1
    z_dense = np.linspace(z_low, z_high, n_dense)

    H0 = _get_param(theta, params, "H0")
    H_dense = H_array(z_dense, theta, params, model)
    E_dense = H_dense / H0
    
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(z_dense, E_dense)
    dE_dz = cs(z_array, 1)
    Ez = cs(z_array, 0)

    return -1.0 + (1.0 + z_array) * dE_dz / Ez

@register_warmup
def warmup():
    params_lcdm = ["H0", "omega_m"]
    theta_lcdm  = np.array([70.0, 0.3])
 
    params_ug   = ["H0", "omega_m", "x"]
    theta_ug    = np.array([70.0, 0.3, 0.1])
 
    z_test = np.linspace(0.01, 2.0, 40)

    params_csl_a = ["H0", "omega_m", "omega_xi"]
    theta_csl_a  = np.array([70.0, 0.3, 1e-4])

    tt_grid_test = np.linspace(-5, 0, 50)

    params_cos_cosh = ["H0", "omega_m", "w_a", "w_b", "w_c"]
    theta_cos_cosh  = np.array([70.0, 0.3, 0.1, 0.1, 0.1])

    comoving_distance(0.5, theta_lcdm, params_lcdm)
    comoving_distance_array(z_test, theta_lcdm, params_lcdm)
    ad_between_distances(0.5, 1.5, theta_lcdm, params_lcdm)

    comoving_distance(0.5, theta_ug, params_ug, model="UG_barotropic")
    comoving_distance_array(z_test, theta_ug, params_ug, model="UG_barotropic")
    ad_between_distances(0.5, 1.5, theta_ug, params_ug, model="UG_barotropic")

    comoving_distance(0.5, theta_csl_a, params_csl_a, model="UG_CSL")
    comoving_distance_array(z_test, theta_csl_a, params_csl_a, model="UG_CSL")
    ad_between_distances(0.5, 1.5, theta_csl_a, params_csl_a, model="UG_CSL")


    comoving_distance(0.5, theta_cos_cosh, params_cos_cosh, model="cos_cosh")
    comoving_distance_array(z_test, theta_cos_cosh, params_cos_cosh, model="cos_cosh")
    ad_between_distances(0.5, 1.5, theta_cos_cosh, params_cos_cosh, model="cos_cosh")
 
    print("cosmology.py: Numba kernels warmed up")

if __name__ == "__main__":
    warmup()