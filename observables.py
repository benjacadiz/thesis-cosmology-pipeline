from cosmology import comoving_distance_array, H_array, angular_diameter_distance, ad_between_distances
from utils import register_warmup
import numpy as np
from numba import njit
from typing import Any

def mu_theoretical(z_arr : np.ndarray[Any, np.dtype[np.float64]], theta : np.ndarray, params: list, model : str = "flat_LCDM" ) -> np.ndarray[Any, np.dtype[np.float64]]:
    distances = comoving_distance_array(z_arr, theta, params, model)
    return _mu_numba(z_arr, distances)

def Hz_theoretical(z_arr : np.ndarray[Any, np.dtype[np.float64]], theta : np.ndarray, params: list, model : str = "flat_LCDM") -> np.ndarray[Any, np.dtype[np.float64]]:
    return H_array(z_arr, theta, params, model)

def Ddt_theoretical(z_l: float, z_s: float, theta: np.ndarray, params: list, model: str = "flat_LCDM") -> float:
    dd = angular_diameter_distance(z_l, theta, params, model)
    ds = angular_diameter_distance(z_s, theta, params, model)
    dds = ad_between_distances(z_l, z_s, theta, params, model)
    return (1.0 + z_l) * dd * ds / dds

def Dd_theoretical(z_l: float, theta: np.ndarray, params: list, model: str = "flat_LCDM") -> float:
    return angular_diameter_distance(z_l, theta, params, model)

def Da_theoretical(z: float, theta: np.ndarray, params: list, model: str = "flat_LCDM") -> float:
    return angular_diameter_distance(z, theta, params, model)
@njit(fastmath=True)
def _mu_numba(z_arr : np.ndarray[Any, np.dtype[np.float64]], dc_array : np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    n_z = z_arr.size
    mu_array = np.empty(n_z)
    for k in range(n_z):
        d_l = (1.0 + z_arr[k]) * dc_array[k]
        mu_array[k] = 5.0*np.log10(d_l) + 25.0
    return mu_array

@register_warmup
def warmup():
    params_lcdm = ["H0", "Omega_m"]
    theta_lcdm = np.array([70.0, 0.3])

    params_ug = ["H0", "Omega_m", "x"]
    theta_ug = np.array([70.0, 0.3, -0.5])


    params_csl_a = ["H0", "Omega_m", "omega_xi"]
    theta_csl_a  = np.array([70.0, 0.3, 1e-4])
    z_test = np.linspace(0.01, 2.0, 40)

    mu_theoretical(z_test, theta_lcdm, params_lcdm)
    Hz_theoretical(z_test, theta_lcdm, params_lcdm)
    mu_theoretical(z_test, theta_ug, params_ug, model="UG_barotropic")
    Hz_theoretical(z_test, theta_ug, params_ug, model="UG_barotropic")
    mu_theoretical(z_test, theta_csl_a, params_csl_a, model="UG_CSL")
    Hz_theoretical(z_test, theta_csl_a, params_csl_a, model="UG_CSL")

    print("observables.py: Numba kernels warmed up")

if __name__ == "__main__":
    warmup()