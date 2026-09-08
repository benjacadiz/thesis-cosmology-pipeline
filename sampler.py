import os
from functools import partial
from collections.abc import Callable
import numpy as np
import emcee


def make_log_probability(
    params: list[str],
    datasets: dict,
    model: str = "flat_LCDM",
    log_posterior_fn: Callable | None = None
) -> Callable:
    """
    Builds the one-argument callable expected by emcee.

    emcee only varies theta. The cosmological parameter names, datasets and
    model are fixed metadata for a given run and must be captured in the
    likelihood callable.

    Parameters
    ----------
    params : list of str
        List of parameter names in the chain.
    datasets : dict
        Dictionary with the datasets used to evaluate the information criteria.
    model : str, optional
        Model name. Present for interface compatibility.
    log_posterior_fn : Callable, optional
        A custom log-posterior function that takes (theta, params, datasets, model) as arguments.
        If None, the default log_posterior function from likelihoods.py will be used.
    
    Returns
    -------
    Callable
        A function that takes a single argument theta (the parameter vector) and returns the log-posterior
        value for that theta given the fixed params, datasets, and model. This function is suitable for use as the
        log-probability function in emcee.
    """
    if log_posterior_fn is None:
        from likelihoods import log_posterior as log_posterior_fn

    return partial(
        log_posterior_fn,
        params=list(params),
        datasets=datasets,
        model=model
    )


def make_filename(model : str, datasets : list[str], params : list[str], directory  : str = "chains") -> str:
    """
    Generates a filename for the MCMC chain based on the model, datasets and parameters used.
    The filename is structured as: {model}_{dataset1}_{dataset2}_..._{param1}_{param2}_...h5
    The file is stored in the specified directory (default "chains"). If the directory does not exist, it will be created.

    Parameters
    ----------
    model : str
        Name of the cosmological model used in the analysis (e.g., "flat_LCDM").
    datasets : list of str
        List of datasets included in the analysis (e.g., ["OHD", "Pantheon+"]).
    params : list of str
        List of cosmological parameters being sampled (e.g., ["H0", "Omega_m"]).
    directory : str, optional
        Directory where the chain file will be stored, by default "chains".

    Returns
    -------
    str
        The generated filename for the MCMC chain.
    """
    os.makedirs(directory, exist_ok=True)
    dataset_str = "_".join(sorted(datasets))
    params_str = "_".join(params)
    filename = f"{model}_{dataset_str}_{params_str}.h5"

    return os.path.join(directory, filename)

def load_or_create_backend(
    filename: str,
    nwalkers: int,
    ndim: int,
    resume: str = "ask",
) -> tuple[emcee.backends.HDFBackend, bool]:
    """
    Checks if a chain file already exists at the specified filename. If it exists and is valid,
    it will load the existing chain and return the backend along with a flag indicating that it is resuming.
    If the file does not exists or is invalid, it will create a new backend for the MCMC sampler and return
    it with a flag indicating that it is not resuming.
    
    Parameters
    ----------
    filename : str
        The path to the chain file to load or create.
    nwalkers : int
        The number of walkers in the MCMC sampler, needed if a new backend is created.
    ndim : int
        The number of dimensions (parameters) in the MCMC sampler, needed if a new backend is created.
    resume : str, optional
        Determines the behavior when a chain file already exists:
        - "ask": Prompt the user to decide whether to resume or overwrite the existing chain. Default behavior.
        - "always": Automatically resume from the existing chain without prompting.
        - "never": Automatically overwrite the existing chain without prompting.
    
    
    Returns
    -------
    backend : emcee.backends.Backend
        The loaded or newly created backend for the MCMC sampler.
    resume : bool
        Flag indicating whether the backend is resuming from an existing chain (True) or is newly created (False).
    """

    if resume not in {"ask", "always", "never"}:
        raise ValueError("resume debe ser 'ask', 'always' o 'never'.")
    

    backend = emcee.backends.HDFBackend(filename)
    if os.path.exists(filename):
        try:
            n_steps = backend.iteration
            if n_steps > 0:
                print(f"Cadena existente encontrada: {filename}")
                print(f"Pasos guardados: {n_steps}")
                print(f"Walkers: {backend.shape[0]}")
                print(f"Número de parámetros: {backend.shape[1]}")

                if backend.shape != (nwalkers, ndim):
                    print(
                        "La forma de la cadena no coincide con la configuración actual: "
                        f"{backend.shape} != {(nwalkers, ndim)}"
                    )
                    resp = input("Sobreescribir cadena existente? [y/n]: ").strip().lower()
                    if resp == "y":
                        print("Sobreescribiendo cadena existente.")
                        backend.reset(nwalkers, ndim)
                        return backend, False
                    else:
                        print("No se sobreescribirá la cadena. Revisar configuración de nwalkers y ndim.")
                        return None, False
                    

                if resume == "always":
                    resp = "y"
                elif resume == "never":
                    resp = "n"
                else:
                    resp = input("Reanudar sampleo? [y/n]: ").strip().lower()

                if resp == "y":
                    print(f"Reanudando desde paso {n_steps}.")
                    return backend, True
                else:
                    print("Sobreescribiendo cadena existente.")
                    backend.reset(nwalkers, ndim)
                    return backend, False
            else:
                backend.reset(nwalkers, ndim)
                return backend, False
        except Exception as exc:
            raise RuntimeError(f"Archivo existente corrupto o incompatible: {filename}") from exc
    else:
        print(f"Creando nueva cadena: {filename}")
        backend.reset(nwalkers, ndim)
        return backend, False

def initialize_walkers(
    p0_center: np.ndarray,
    p0_sigma: np.ndarray | float,
    nwalkers: int,
    ndim: int,
    log_probability: Callable,
    max_tries: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Initializes the positions of the MCMC walkers in a Gaussian ball around a central point (p0_center)
    with a specified standard deviation (p0_sigma). The function checks that the initial positions of the
    walkers have finite log-posterior values before accepting them. If a valid position is not found for a
    walker after a certain number of attempts (max_tries), an error is raised.
    
    Parameters
    ----------
    p0_center : np.ndarray
        The central point around which the walkers will be initialized, with shape (ndim,).
    p0_sigma : np.ndarray
        The standard deviation for the Gaussian distribution used to initialize the walkers, with shape (ndim,).
    nwalkers : int
        The number of walkers to initialize.
    ndim : int
        The number of dimensions (parameters) in the MCMC sampler.
    log_probability : Callable
        A function that takes a parameter vector and returns the log-posterior value, used to check the validity of the initial positions.
    max_tries : int, optional
        The maximum number of attempts to find valid initial positions for all walkers, by default 100.
    rng: np.random.Generator, optional
        A random number generator instance to use for generating the initial positions. If None, a new
        default_rng() will be created.
        
    Returns
    -------
    pos0 : np.ndarray
        An array of shape (nwalkers, ndim) containing the initialized positions of the walkers"""
    p0_center = np.asarray(p0_center, dtype=float)
    p0_sigma = np.asarray(p0_sigma, dtype=float)

    if p0_center.shape != (ndim,):
        raise ValueError(f"p0_center debe tener forma {(ndim,)}, recibido {p0_center.shape}.")
    if p0_sigma.ndim == 0:
        p0_sigma = np.full(ndim, float(p0_sigma))
    if p0_sigma.shape != (ndim,):
        raise ValueError(f"p0_sigma debe ser escalar o tener forma {(ndim,)}, recibido {p0_sigma.shape}.")
    if np.any(p0_sigma <= 0.0):
        raise ValueError("Todos los anchos iniciales p0_sigma deben ser positivos.")
    if nwalkers < 2 * ndim:
        raise ValueError(
            f"emcee recomienda nwalkers >= 2*ndim. Recibido nwalkers={nwalkers}, ndim={ndim}."
        )

    rng = np.random.default_rng() if rng is None else rng

    # Genera posiciones iniciales en una nube gaussiana y conserva solo
    # walkers con posterior finito.
    pos0 = np.zeros((nwalkers, ndim))
    n_valid = 0
    attempts = 0

    while n_valid < nwalkers and attempts < max_tries:
        n_missing = nwalkers - n_valid
        candidates = p0_center + p0_sigma * rng.normal(size=(2 * n_missing, ndim))
        for c in candidates:
            if n_valid >= nwalkers:
                break
            lp = log_probability(c)
            if np.isfinite(lp):
                pos0[n_valid] = c
                n_valid +=1
        attempts +=1

    if n_valid < nwalkers:
        raise RuntimeError(
            f"Sólo se encontraron {n_valid}/{nwalkers} walkers válidos "
            f"tras {max_tries} intentos. Revisar p0_center, p0_sigma y "
            "los rangos del prior"
        )
    
    print(f"Walkers inicializados: {n_valid}/{nwalkers} válidos")
    return pos0


def run(
    log_probability: Callable,
    pos0: np.ndarray | None,
    nsteps: int,
    backend: emcee.backends.Backend,
    resume: bool = False,
    progress: bool = True,
    moves=None,
    pool = None,
    skip_initial_state_check: bool = False,
) -> emcee.EnsembleSampler:
    
    """Wrapper to run the MCMC sampler with progress bar and optional resuming if a chain already exists in the backend.
    The funcion checks the shape of pos0 and the backend to ensure they are compatible, and handles the logic for resuming
    or starting a new chain. If resuming, it calculates how many steps are remaining to reach nsteps and runs the sampler for
    those steps.
    Allows passing custom moves and a multiprocessing pool to the sampler.
    
    Parameters
    ----------
    log_probability : Callable
        A function that takes a parameter vector and returns the log-posterior value, used by the MCMC sampler.
    pos0 : np.ndarray or None
        An array of shape (nwalkers, ndim) containing the initial positions of the walkers. If None, the function will
        attempt to resume from the backend.
    nsteps : int
        The total number of MCMC steps to run (including any already present in the backend if resuming).
    backend : emcee.backends.Backend
        The backend to use for storing the MCMC chain. This should be an instance of emcee.backends.HDFBackend or similar.
    resume : bool, optional
        Whether to attempt to resume from an existing chain in the backend if pos0 is None, by default False.
    progress : bool, optional
        Whether to display a progress bar during sampling, by default True.
    moves : emcee.moves.Move or list of emcee.moves.Move, optional
        Custom move strategy or list of strategies to use in the sampler, by default None (which uses the default moves).
    pool : multiprocessing.Pool or None, optional
        A multiprocessing pool to use for parallelizing the likelihood evaluations, by default None (no parallelism).
    skip_initial_state_check : bool, optional
        Whether to skip the initial state check when resuming from a backend, by default False.
        If True, the sampler will not check that the initial state of the walkers in the backend is valid before resuming.
    
    Returns
    -------
    sampler : emcee.EnsembleSampler
        The MCMC sampler instance after running the specified number of steps.
    """

    # Wrapper para correr sampler con progress_bar
    if resume and pos0 is None:
        nwalkers, ndim = backend.shape
    elif pos0 is not None:
        nwalkers, ndim = pos0.shape
        if resume and backend.iteration > 0 and backend.shape != (nwalkers, ndim):
            raise ValueError(
                f"La forma de pos0 {(nwalkers, ndim)} no coincide con el backend {backend.shape}."
            )
    else:
        raise ValueError("pos0 no puede ser None al iniciar una cadena nueva.")

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        backend=backend,
        moves=moves,
        pool = pool
    )
    if backend.iteration >= nsteps:
        print(f"Cadena ya completa con {nsteps} pasos. No se correrá el sampler.")
        return sampler
    if resume:
        pos0 = None
        n_prev = backend.iteration
        print(f"Reanudando desde paso {n_prev}...")
    else:
        n_prev = 0
        print(f"Comenzando sampleo de {nsteps} pasos con {nwalkers} walkers...")

    n_remaining = nsteps - n_prev
    sampler.run_mcmc(
        pos0,
        n_remaining,
        progress=progress,
        skip_initial_state_check=skip_initial_state_check, store=True
    )
    n_total = backend.iteration
    print(f"Sampleo completado: {n_total} pasos guardados en {backend.filename}")
    return sampler

def get_samples(sampler : emcee.EnsembleSampler, discard : int, thin : int = 1) -> np.ndarray:
    """
    Extracts the MCMC samples from the sampler after discarding the burn-in steps and applying thinning.
    The samples are returned flattened across all walkers, resulting in an array of shape (n_samples, ndim) where n_samples
    is the total number of samples after discarding and thinning.
    
    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The MCMC sampler instance from which to extract the samples. It is assumed that the sampler has already been run and contains
        the MCMC chain in its backend.
    discard : int
        The number of initial steps to discard as burn-in. This should be a non-negative integer less than the total number of steps in the sampler.
    thin : int, optional
        The thinning factor to apply to the chain. Only every 'thin'-th sample will be returned after discarding the burn-in steps.
    
    Returns
    -------
    samples : np.ndarray
        An array of shape (n_samples, ndim) containing the MCMC samples after discarding the burn-in steps and applying thinning.

    """

    #Extraer muestras del sampler, post burn-in aplanadas.
    nsteps = sampler.iteration
    if discard >= nsteps:
        raise ValueError(f"Discard ({discard}) debe ser menor que el número de pasos ({nsteps}).")
    
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    print(f"Muestras extraídas: {len(samples)}")
    print(f"Burn-in descartado: {discard} pasos")
    print(f"Thinning aplicado: cada {thin} pasos")

    return samples

def check_convergence(
    sampler: emcee.EnsembleSampler,
    params: list[str] | None = None,
    tol: float = 50.0,
) -> np.ndarray | None:
    """
    Calculate the autocorrelation time of the MCMC chain and check if the number of steps is sufficient for convergence.
    The function prints the total number of steps, the mean acceptance fraction and the autocorrelation time for each
    parameter. It also checks if the ratio of the total steps to the autocorrelation time is greater than a specified
    tolerance level (tol) for each parameter.
    If the autocorrelation time cannot be estimated (e.g., if the chain is too short), it catches the exception and prints 
    a warning message.
    It returns the estimated autocorrelation time if successful, or None if it cannot be estimated.
    
    Parameters
    ----------
    sampler: emcee.EnsembleSampler
        The MCMC sampler instance from which to calculate the autocorrelation time. It is assumed that the sampler
        has already been run and contains the MCMC chain in its backend.
    params: list of str or None, optional
        A list of parameter names corresponding to the dimensions in the MCMC sampler, used for labeling the output.
        If None, the parameters will be labeled generically as 'param[i]'. By default, None.
    tol: float, optional
        The tolerance level for the ratio of total steps to autocorrelation time to consider the chain as converged.
        If the ratio nsteps/tau is greater than tol for all parameters, the chain is considered to have sufficient steps for convergence.
        By default, tol=50.0.
    
    Returns
    -------
    tau : np.ndarray or None
        An array containing the estimated autocorrelation time for each parameter if successful, or None if it cannot be estimated.
    
    """
    #Calcular tiempo de autocorrelación y verificar que se hayan corrido al menos 50 veces ese tiempo.
    nsteps = sampler.iteration
    if hasattr(sampler, "acceptance_fraction"):

        accept = np.mean(sampler.acceptance_fraction)
        print(f"Tasa de aceptación media: {accept:.3f}")
    else:
        print("Tasa de aceptación no disponible desde backend")
    print(f"Pasos totales: {nsteps}")

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"Tiempo de autocorrelación: {tau}")
        for i, t in enumerate(tau):
            label = params[i] if params is not None else f"param[{i}]"
            ratio = nsteps / t
            status = "ok" if ratio > tol else "Cadena muy corta"
            print(f"  {label}: {t:.1f} pasos, n/t = {ratio:.1f} -> {status}")

        tau_max = np.max(tau)
        if hasattr(sampler, "nwalkers"):
            n_eff = nsteps * sampler.nwalkers / tau_max
        else:
            nwalkers = sampler.get_chain().shape[1] if sampler.get_chain().ndim == 3 else 1
            n_eff = nsteps * nwalkers / tau_max
        print(f"Muestras efectivas aproximadas: {n_eff:.1f}")
        return tau
    
    except emcee.autocorr.AutocorrError:
        print("No fue posible estimar tiempo de autocorrelación. Posiblemente la cadena es muy corta.")
        print(f"Considerar correr hasta al menos ~{int(2*nsteps)} pasos y volver a evaluar.")
        return None

def get_bestfit(samples: np.ndarray, log_probs: np.ndarray, params: list[str], sigma:float = 1.0) ->dict[str, float]:
    """
    Calculates the best-fit parameters and their confidence intervals from the MCMC samples and log-probabilities.
    The function identifies the maximum a posteriori (MAP) parameters and their log-probability. Computes the median
    and confidence intervals corresponding to the specified sigma for each parameter.
    
    Parameters
    ----------
    samples : np.ndarray
        An array of shape (n_samples, ndim) containing the MCMC samples of the parameters.
    log_probs : np.ndarray
        An array of shape (n_samples,) containing the log-probability values corresponding to each sample in the 'samples' array.
        This is typically obtained from sampler.get_log_prob(discard=..., flat=True) after running the MCMC sampler.
    params : list of str
        A list of parameter names corresponding to the dimensions in the 'samples' array, used for labeling the output.
    sigma : float, optional
        The number of standard deviations for the confidence intervals (e.g., 1.0 for 68% confidence), by default 1.0.
        Only values of 1, 2, or 3 are accepted, corresponding to the standard Gaussian quantiles.
        
    Returns
    -------
    results : dict
        A dictionary containing the best-fit results, including:
        - "median": The median values of the parameters across the samples.
        - "low": The lower bounds of the confidence intervals for each parameter.
        - "high": The upper bounds of the confidence intervals for each parameter.
        - "map": The parameter values corresponding to the maximum a posteriori (MAP) estimate.
        - "lp_map": The log-probability value at the MAP estimate.
        - "chi2_min": The minimum chi-squared value corresponding to the MAP estimate, calculated as -2 * lp_map.
        - "sigma": The sigma level used for the confidence intervals.

    """
    sigma_to_quantile = {1 : 0.3173, 2 : 0.0455, 3 : 0.0027}
    if sigma not in sigma_to_quantile:
        raise ValueError(f"sigma debe ser uno 1, 2 o 3. Recibido {sigma}")
    
    alpha = sigma_to_quantile[sigma]
    q_low = 100 * alpha / 2
    q_high = 100 * (1 - alpha / 2)

    median = np.median(samples, axis=0)
    low = np.percentile(samples, q_low, axis=0)
    high = np.percentile(samples, q_high, axis=0)


    idx_map = np.argmax(log_probs)
    map_vals = samples[idx_map]
    lp_map = float(log_probs[idx_map])

    results = {"median" : median,
               "low" : low,
               "high" : high,
               "map" : map_vals,
               "lp_map" : lp_map,
               "chi2_min" : -2*lp_map,
               "sigma" : sigma}
    
    print(f"Best fit ({sigma}σ interval):")
    print(f"{'-'*40}")
    for k, name in enumerate(params):
        print(f"{name:10s} : {median[k]:.4f} [-{np.abs(median[k] - low[k]):.4f}, +{np.abs(high[k] - median[k]):.4f}]")
    print(f"{'-'*40}")
    print(f"MAP: {map_vals} (log-posterior: {lp_map:.2f}, chi2_min: {results['chi2_min']:.2f})")
    print(f"{'-'*40}")
    return results   