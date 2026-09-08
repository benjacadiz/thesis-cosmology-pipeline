import numpy as np
import emcee
from getdist import MCSamples
from likelihoods import log_likelihood_combined
from sampler import check_convergence, get_bestfit
import os
from scipy.integrate import quad

mapping = {"h0": r"H_0",
           "omega_m": r"\Omega_m",
           "x": r"x",
           "u0": r"u_0",
           "xi_csl" : r"\xi_{\mathrm{CSL}}",
           "omega_xi" : r"\Omega_{\xi}",
           "rd" : r"r_d",
           "omega_lambda" : r"\Omega_{\Lambda_{\rm eff}}",
           "wb" : r"\Omega_b h^2",
           "xi_tilde" : r"\tilde{\xi}",
           "omega_b0" : r"\Omega_{b0}",
           "omega_dm0" : r"\Omega_{dm0}",
           }


def analyze(backend_path: str, params: list, datasets: dict, model: str = "flat_LCDM",
            thin: int = 1, labels: list = None, discard_fallback: int = 300) -> dict:
    """
    Load an MCMC backend, apply burn-in and thinning, and compute summary
    statistics for the sampled parameters.

    Parameters
    ----------
    backend_path : str
        Path to the emcee HDF backend file.
    params : list
        List of parameter names in the chain.
    datasets : dict
        Dictionary with the datasets used to evaluate the information criteria.
    model : str, optional
        Model name. Present for interface compatibility.
    thin : int, optional
        Thinning factor applied to the chain.
    labels : list, optional
        Custom labels for the parameters. If None, LaTeX labels are generated.
    discard_fallback : int, optional
        Number of initial steps to discard if convergence time cannot be estimated.

    Returns
    -------
    dict
        Dictionary containing GetDist samples, best-fit values, AIC/BIC,
        convergence information, and the backend object.
    """
    backend = emcee.backends.HDFBackend(backend_path, read_only = True)
    nsteps, nwalkers, ndim = backend.get_chain().shape
    print(f"Cadena cargada con {nwalkers} walkers, {ndim} parámetros, y {nsteps} pasos.")
    tau = check_convergence(backend, params=params)
    if tau is not None:
        discard = int(2*np.max(tau))
        print(f"Descartando los primeros {discard} pasos como burn-in.")
    else:
        discard = discard_fallback
        print(f"No se pudo estimar tau, usando un valor de descarte por defecto de {discard} pasos.")

    if discard >= nsteps:
        raise ValueError(f"El número de pasos a descartar ({discard}) es mayor o igual al número total de pasos ({nsteps}). "
                         f"Considerar aumentar número de pasos o reducir el valor de descarte.")
    
    samples = backend.get_chain(discard=discard, thin=thin, flat=True)
    log_probs = backend.get_log_prob(discard=discard, thin=thin, flat=True)
    nsamples = len(samples)
    print(f"Obtenidos {nsamples} muestras después de descartar y hacer thinning.")

    names = params.copy()
    if model == "UG_CSL":
        xi_samples = omegaxi_to_xi(samples, params)
        samples = np.hstack((samples, xi_samples[:, np.newaxis]))
        names.append("xi_csl")
        omega_l_samples = compute_omegal(samples, params)
        samples = np.hstack((samples, omega_l_samples[:, np.newaxis]))
        names.append("omega_lambda")
    if model == "UG_CSL_baryon":
        xitilde_samples = samples[:, params.index("xi_tilde")]
        xi_samples = xitilde_samples * samples[:, params.index("H0")] / 3.085e19
        samples = np.hstack((samples, xi_samples[:, np.newaxis]))
        names.append("xi_csl")
        omega_l_samples = 1 - samples[:, params.index("Omega_dm0")] - samples[:, params.index("Omega_b0")]
        samples = np.hstack((samples, omega_l_samples[:, np.newaxis]))
        names.append("omega_lambda")
    if "DESI" in datasets.keys():
        rd_samples = compute_rd(samples, params, model=model)
        samples = np.hstack((samples, rd_samples[:, np.newaxis]))
        names.append("rd")
    
    if labels is None:
        labels = _string_to_LaTex(names)

    gd_samples = MCSamples(samples=samples, names=names, labels=labels)

    bestfit = get_bestfit(samples, log_probs, params)

    k = len(params)
    N = _count_datapoints(datasets)
    ic = aic_bic(bestfit["chi2_min"], k, N)

    print(f"\n{'-'*40}")
    print(f"{'AIC:':<10} {ic['aic']:.2f}")
    print(f"{'BIC:':<10} {ic['bic']:.2f}")
    print(f" k  = {k}, N = {N}")
    print(f"{'-'*40}")

    return {"gd_samples" : gd_samples,
            "bestfit" : bestfit,
            "aic_bic" : ic["aic"],
            "bic" : ic["bic"],
            "tau" : tau,
            "discard" : discard,
            "nsamples" : nsamples,
            "backend" : backend}


def compute_rd(samples, params, Neff = 3.044, model = "flat_LCDM"):
    params_low = [p.lower() for p in params]
    H0 = samples[:, params_low.index("h0")]
    h = H0 / 100.0
    if model in ("flat_LCDM", "UG_CSL", "UG_barotropic"):
        omega_m = samples[:, params_low.index("omega_m")]
        omega_m_h2 = omega_m * h**2
        wb = samples[:, params_low.index("wb")]

    elif model in ("UG_CSL_baryon"):
        omega_b = samples[:, params_low.index("omega_b0")]
        omega_dm = samples[:, params_low.index("omega_dm0")]
        omega_m = omega_b + omega_dm
        omega_m_h2 = omega_m * h**2
        wb = omega_b * h**2
    
    n = len(samples)
    if wb is None:
        rng = np.random.default_rng(seed = 42)
        wb = rng.normal(0.02218, 0.00055, size=n)
    return 147.05 * (wb / 0.02236)**(-0.13) * (omega_m_h2 / 0.1432)**(-0.23) * (Neff / 3.04)**(-0.1)

def omegaxi_to_xi(samples, params):
    params_low = [p.lower() for p in params]
    if "omega_xi" not in params_low:
        raise ValueError("Para calcular xi a partir de Omega_xi se necesita el parámetro 'Omega_xi' en la cadena.")
    omega_xi = samples[:, params_low.index("omega_xi")]
    H0 = samples[:, params_low.index("h0")]

    return 3.0*H0 *  np.sqrt(omega_xi) * 3.2408e-20

def compute_omegal(samples, params):
    params_low = [p.lower() for p in params]
    if "omega_m" not in params_low:
        raise ValueError("Para calcular Omega_lambda a partir de Omega_m se necesita el parámetro 'Omega_m' en la cadena.")
    omega_m = samples[:, params_low.index("omega_m")]
    omega_xi = samples[:, params_low.index("omega_xi")] if "omega_xi" in params_low else 0.0
    return 1.0 - omega_m - 2*np.sqrt(omega_xi)

def compute_age(bestfit: dict[str, float], model = "flat_LCDM" ):
    H0 = bestfit["H0"]
    if model == "flat_LCDM":
        omega_m = bestfit["Omega_m"]
        Ez = lambda z: np.sqrt(omega_m * (1 + z)**3 + (1 - omega_m))
    elif model == "UG_barotropic":
        omega_m = bestfit["Omega_m"]
        x = bestfit["xi"]
        Ez = lambda z: np.sqrt(omega_m * (1 + x) * (1 + z)**(3/(1 + x)) + (1 - omega_m - x * omega_m))
    elif model == "UG_CSL":
        pass
    elif model == "UG_CSL_baryon":
        pass
    else:
        raise ValueError(f"Modelo desconocido: {model}")
    integrand = lambda z: 1.0 / ((1.0+ z) *Ez)
    age, err = quad(integrand, 0, np.inf)
    age_Gyr = age / (H0 * 3.2408e-20) / (3.1536e16 * 1e9)
    return age_Gyr
    

def aic_bic(chi2_min, k, N):
    aic = chi2_min + 2 * k
    bic = chi2_min + k * np.log(N)
    return {"aic" : float(aic),
            "bic" : float(bic)}

def _string_to_LaTex(params: list[str])-> list[str]:

    return [mapping.get(p.lower(), p) for p in params]

def _count_datapoints(datasets):
    count = 0
    for name, data in datasets.items():
        name = name.lower()
        if name in ("ohd", "cc"):
            count += len(data["z"])
        elif name in ("pantheon", "pantheon+"):
            count += len(data["z"])
        elif name == "desi":
            count += sum(len(t["obs"]) for t in data["tracers"])
        elif name == "h0licow":
            count += len(data)
        elif name == "bhs":
            count += len(data["z"])
        else:
            raise ValueError(f"Unknown dataset name: {name}")
    return count

