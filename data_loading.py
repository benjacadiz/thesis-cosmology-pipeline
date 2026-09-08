import numpy as np
import os
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator, interp1d
import joblib
_H0LICOW_FILES = {
    "B1608"  : "B1608_Dd_Ddt_params.dat",
    "HE0435" : "HE0435_Ddt_AO+HST.dat",
    "J1206"  : "J1206_final.csv",
    "PG1115" : "PG1115_AO+HST_Dd_Ddt.dat",
    "RXJ1131": "RXJ1131_AO+HST_Dd_Ddt.dat",
    "WFI2033": "wfi2033_dt_bic.dat",
    "redshifts": "H0LiCOW_redshifts.txt",
}



def load_ohd(filepath : str, zmin: float = 0.0 , zmax : float = np.inf) -> dict[str, np.ndarray]:
    """ 
    Loads data for Observational Hubble Data (OHD) from .txt file. The file is expected to have
    named columns for redshift z, Hubble parameter H(z) and its error sigmaH. The function
    admits zmin and zmax parameters to filter the data by redshift range, by default includes all data.
    Also checks the integrity of the data and prints a summary of the loaded dataset.

    It will return a dictionary with keys "z", "Hz" and "sigmaH" containing the corresponding data as numpy arrays.

    
    Parameters
    ----------
    filepath : str
        Path to the .txt file containing the OHD data.
    zmin : float, optional
        Minimum redshift to include in the dataset, by default 0.0
    zmax : float, optional
        Maximum redshift to include in the dataset, by default np.inf

    Returns
    -------
    data : dict
        Dictionary containing the loaded OHD data.

    Raises
    ------
    IOError
        If there is an error reading the file.
    
    """
  
    _check_file(filepath)

    try:
        raw = np.genfromtxt(filepath, names=True, encoding="utf-8")
    except Exception as e:
       raise IOError(f"Error leyendo OHD desde {filepath} : {e}")
    
    z = raw["z"].astype(np.float64)
    Hz = raw["Hz"].astype(np.float64)
    sigmaH = raw["sigmaH"].astype(np.float64)

    mask = (z >= zmin) & (z <= zmax) & np.isfinite(Hz) & (sigmaH>0)
    z, Hz, sigmaH = z[mask], Hz[mask], sigmaH[mask]

    data = {"z": z.astype(np.float64),
            "Hz": Hz.astype(np.float64),
            "sigmaH": sigmaH.astype(np.float64)}
    
    _validate(data, "OHD", required = ["z", "Hz", "sigmaH"])
    summarize(data, "OHD")
    return data

def load_pantheon(filepath_dat : str, filepath_cov : str, zmin : float =0.0, zmax : float = np.inf, fitprob_min : float = -1.0) -> dict[str, np.ndarray]:
    """
    Loads data for the Pantheon+ supernova compilation and its covariance matrix.
    When loading the data, it applies quality cuts based on redshift range and fit probability.
    Checks the integrity of the data and prints a summary of the loaded dataset.
    
    It also loads the covariance matrix from the .cov file, checks that its size matches the number of
    SNe after cuts, and computes its inverse for later use in likelihood calculations.
    
    Returns a dictionary with keys 'z', 'm_b_corr', 'cov', 'cov_inv', 'IS_CALIBRATOR' and 'CEPH_DIST'
    containing the corresponding data as numpy arrays.
    
    Parameters
    ----------
    filepath_dat : str
        Path to the .dat file containing the Pantheon+ data.
    filepath_cov : str
        Path to the .cov file containing the covariance matrix for the Pantheon+ data.
    zmin : float, optional
        Minimum redshift to include in the dataset, by default 0.0
    zmax : float, optional
        Maximum redshift to include in the dataset, by default np.inf
    fitprob_min : float, optional
        Minimum fit probability to include a supernova in the dataset, by default -1.0 (no cut)
        
    Returns
    -------
    data : dict
        Dictionary containing the loaded Pantheon+ data and covariance information.
    """
    _check_file(filepath_dat)
    _check_file(filepath_cov)
   
    raw = np.genfromtxt(filepath_dat, names=True, dtype=None, encoding="utf-8")
    z = raw["zHD"].astype(np.float64)
    m_b_corr = raw["m_b_corr"].astype(np.float64)
    is_calib = raw["IS_CALIBRATOR"].astype(np.int32)
    ceph_dist = raw["CEPH_DIST"].astype(np.float64)
    fitprob = raw["FITPROB"].astype(np.float64)

    N_total = len(z)
    mask = _quality_mask(z, fitprob, is_calib, zmin, zmax, fitprob_min)
    n_cut = N_total - mask.sum()

    z = z[mask]
    m_b_corr = m_b_corr[mask]
    is_calib = is_calib[mask]
    ceph_dist = ceph_dist[mask]

    ceph_dist = np.where(is_calib, ceph_dist, 0.0)
    cov = _read_cov(filepath_cov, N_total)[np.ix_(mask, mask)]
    cov_inv = np.linalg.inv(cov)

    data = {"z" : z.astype(np.float64),
            "m_b_corr" : m_b_corr.astype(np.float64),
            "cov" : cov,
            "cov_inv" : cov_inv,
            "IS_CALIBRATOR" : is_calib,
            "CEPH_DIST" : ceph_dist.astype(np.float64)}

    _validate(data, "Pantheon", required=["z", "m_b_corr", "cov_inv", "IS_CALIBRATOR", "CEPH_DIST"])
    summarize(data, "Pantheon", n_cut=n_cut)
    return data

def load_desi(filepath : str, filepath_cov : str = None) -> dict[str, np.ndarray]:

    _check_file(filepath)

    raw = np.genfromtxt(filepath, names=True, dtype=None, encoding="utf-8",
                         missing_values="ND", filling_values=np.nan)
    tracers = []
    for row in raw:
        name = str(row["qTracer"]).strip()
        zeff = float(row["zeff"])
        DVrd = float(row["DVrd"])
        sigmaDVrd = float(row["sigmaDVrd"])
        DMrd = float(row["DMrd"])
        sigmaDMrd = float(row["sigmaDMrd"])
        DHrd = float(row["DHrd"])
        sigmaDHrd = float(row["sigmaDHrd"])

        has_DV = np.isfinite(DVrd)
        has_2D = np.isfinite(DMrd) and np.isfinite(DHrd)

        if has_DV:
            tracer = {"name": name,
                       "zeff": zeff,
                       "type" : "dv",
                       "obs" : np.array([DVrd]),
                       "sigma" : np.array([sigmaDVrd])}
        elif has_2D:
            tracer = {"name": name,
                       "zeff": zeff,
                       "type" : "dm_dh",
                       "obs" : np.array([DMrd, DHrd]),
                       "sigma" : np.array([sigmaDMrd, sigmaDHrd])}
        else:
            raise ValueError(f"Tracers {name} en DESI no tiene datos válidos ni para DV ni para DM/DH")
        tracers.append(tracer)
    
    for t in tracers:
        t["cov_inv"] = np.diag(1.0 / t["sigma"]**2)

    data = {"tracers" : tracers, "rd" :None}
    _summarize_desi(data)
    return data

def load_h0licow(dirpath : str):
    redshifts = _load_redshifts(os.path.join(dirpath, _H0LICOW_FILES["redshifts"]))

    lenses = {}

    lenses["B1608"] = _load_lens_sln_params(os.path.join(dirpath, _H0LICOW_FILES["B1608"]),
                                     redshifts["B1608"])
    
    lenses["HE0435"] = _load_lens_hist1d(os.path.join(dirpath, _H0LICOW_FILES["HE0435"]),
                                        redshifts["HE0435"])
    
    lenses["J1206"] = _load_lens_kde2d_full(os.path.join(dirpath, _H0LICOW_FILES["J1206"]),
                                      redshifts["SDSS1206"])
    
    lenses["PG1115"] = _load_lens_hist2d(os.path.join(dirpath, _H0LICOW_FILES["PG1115"]),
                                       redshifts["PG1115"], col_dd=0, col_ddt=1, delimiter=None)
    
    lenses["RXJ1131"] = _load_lens_hist2d(os.path.join(dirpath, _H0LICOW_FILES["RXJ1131"]),
                                         redshifts["RXJ1131"], col_dd=0, col_ddt=1, delimiter=None)
    
    lenses["WFI2033"] = _load_lens_hist1d_weighted(os.path.join(dirpath, _H0LICOW_FILES["WFI2033"]),
                                                redshifts["WFI2033"])
    
    _summarize_h0licow(lenses)
    return lenses

def load_bhs(filepath):
    _check_file(filepath)
    raw = np.genfromtxt(filepath, names=True, dtype=None, encoding="utf-8")
    
    names = [str(n).strip() for n in raw["Name"]]
    z = raw["z"].astype(np.float64)
    R = raw["R"].astype(np.float64)
    dR = raw["dR"].astype(np.float64)
    M = raw["M"].astype(np.float64)
    dM = raw["dM"].astype(np.float64)

    data = {"names" : names,
            "z" : z,
            "R" : R,
            "dR" : dR,
            "M" : M,
            "dM" : dM}
    _summarize_bhs(data)
    return data


def _load_redshifts(filepath):
    _check_file(filepath)
    raw = np.genfromtxt(filepath, names=True, dtype=None, encoding="utf-8", delimiter="\t")
    redshifts = {}
    for row in raw:
        name = str(row["Lens_Name"]).strip()
        redshifts[name] = {"zl" : float(row["zl"]), "zs" : float(row["zs"])}

    return redshifts

def _summarize_h0licow(lenses):
    print(f"\n{'─'*45}")
    print("Dataset: H0LiCOW")
    print(f"{'─'*45}")
    for name, lens in lenses.items():
        print(f"  Lente {name}: zl={lens['zl']:.3f}, zs={lens['zs']:.3f}, tipo={lens['type']}")
    print(f"{'─'*45}\n")


def _quality_mask(z, fitprob, is_calib, zmin, zmax, fitprob_min):
    mask_z = (z >= zmin) & (z <= zmax)
    mask_fitprob = fitprob > fitprob_min
    mask_calib = is_calib.astype(np.bool_)

    return mask_calib | (mask_z & mask_fitprob)

def _check_file(filepath):
   if not os.path.exists(filepath):
       raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

def _read_cov(filepath_cov, n_expected):
    with open(filepath_cov, "r") as f:
        n = int(f.readline().strip())
        if n != n_expected:
            raise ValueError(f"Tamaño de covarianza ({n}) no coincide con número de SNe en datos ({n_expected})")
        vals = np.array([float(x) for line in f for x in line.split()])
    
    if vals.size != n*n:
        raise ValueError(f"Se esperaban {n*n} valores en cov, encontrados {vals.size }")
    
    return vals.reshape(n,n)

def _build_kde1d(samples, lens_name, weights=None, bandwidth=20, nbins=400, n_grid=1000):
    from sklearn.neighbors import KernelDensity

    counts, edges = np.histogram(samples, bins=nbins, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask = counts > 0
    kde_bins = centers[mask].reshape(-1, 1)
    kde_weights = counts[mask].astype(np.float64)

    if os.path.exists(f"kde1d_{lens_name}.joblib"):
        kde = joblib.load(f"kde1d_{lens_name}.joblib")
    else: 
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(kde_bins, sample_weight=kde_weights)
        joblib.dump(kde, f"kde1d_{lens_name}.joblib")

    grid_low = samples.min() - 3.0*bandwidth
    grid_high = samples.max() + 3.0*bandwidth
    grid = np.linspace(grid_low, grid_high, n_grid)
    log_density = kde.score_samples(grid.reshape(-1, 1))

    interp = interp1d(grid, log_density, kind="linear", bounds_error=False, fill_value=-np.inf)
    return {"interpolator" : interp,
            "ddt_range" : (grid_low, grid_high)}

def _build_kde2d(dd, ddt, lens_name, weights=None, bandwidth=20, nbins=80, n_grid=300):
    from sklearn.neighbors import KernelDensity
    
    hist, dd_edges, ddt_edges = np.histogram2d(dd, ddt, bins=nbins, weights=weights)
    dd_centers = 0.5 * (dd_edges[:-1] + dd_edges[1:])
    ddt_centers = 0.5 * (ddt_edges[:-1] + ddt_edges[1:])

    DD, DDT = np.meshgrid(dd_centers, ddt_centers, indexing="ij")
    counts_flat = np.ravel(hist)
    mask = counts_flat > 0
    kde_bins = np.column_stack((DD.ravel()[mask], DDT.ravel()[mask]))
    kde_weights = counts_flat[mask].astype(np.float64)

    if os.path.exists(f"kde2d_{lens_name}.joblib"):
        kde = joblib.load(f"kde2d_{lens_name}.joblib")
    else:
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(kde_bins, sample_weight=kde_weights)
        joblib.dump(kde, f"kde2d_{lens_name}.joblib")

    dd_low, dd_high = dd.min() - 3.0*bandwidth, dd.max() + 3.0*bandwidth
    ddt_low, ddt_high = ddt.min() - 3.0*bandwidth, ddt.max() + 3.0*bandwidth
    dd_grid = np.linspace(dd_low, dd_high, n_grid)
    ddt_grid = np.linspace(ddt_low, ddt_high, n_grid)
    DD_grid, DDT_grid = np.meshgrid(dd_grid, ddt_grid, indexing="ij")
    grid_points = np.column_stack((DD_grid.ravel(), DDT_grid.ravel()))

    log_density = kde.score_samples(grid_points).reshape(n_grid, n_grid)

    interp = RegularGridInterpolator((dd_grid, ddt_grid), log_density,
                                    method="linear", bounds_error=False, fill_value=-np.inf)
    return {"interpolator" : interp,
            "dd_range" : (dd_low, dd_high),
            "ddt_range" : (ddt_low, ddt_high)}

def _build_kde2d_full(dd, ddt, lens_name, bandwidth=20, n_grid=200):
    from sklearn.neighbors import KernelDensity

    kde_bins = np.column_stack((dd, ddt))
    if os.path.exists(f"kde2d_full_{lens_name}.joblib"):
        kde = joblib.load(f"kde2d_full_{lens_name}.joblib")
    else:
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(kde_bins)
        joblib.dump(kde, f"kde2d_full_{lens_name}.joblib")

    dd_low, dd_high = dd.min() - 3.0*bandwidth, dd.max() + 3.0*bandwidth
    ddt_low, ddt_high = ddt.min() - 3.0*bandwidth, ddt.max() + 3.0*bandwidth
    dd_grid = np.linspace(dd_low, dd_high, n_grid)
    ddt_grid = np.linspace(ddt_low, ddt_high, n_grid)
    DD_grid, DDT_grid = np.meshgrid(dd_grid, ddt_grid, indexing="ij")
    grid_points = np.column_stack((DD_grid.ravel(), DDT_grid.ravel()))

    log_density = kde.score_samples(grid_points).reshape(n_grid, n_grid)

    interp = RegularGridInterpolator((dd_grid, ddt_grid), log_density,
                                    method="linear", bounds_error=False, fill_value=-np.inf)
    
    return {"interpolator" : interp,
            "dd_range" : (dd_low, dd_high),
            "ddt_range" : (ddt_low, ddt_high)}


def _load_lens_hist1d(filepath, zs_dict, bandwith=20, nbins=400):
    _check_file(filepath)

    samples = np.genfromtxt(filepath, comments="#", encoding="utf-8").flatten()
    samples = samples[np.isfinite(samples)]
    lens_name = os.path.splitext(os.path.basename(filepath))[0]
    hist1d = _build_kde1d(samples, lens_name, bandwidth=bandwith, nbins=nbins)

    return {"type" : "hist1d",
            "zl" :  zs_dict["zl"],
            "zs" : zs_dict["zs"],
            "hist1d" : hist1d}

def _load_lens_sln_params(filepath, zs_dict):
    _check_file(filepath)
    raw = np.genfromtxt(filepath, comments="#", encoding="utf-8")

    return {"type" : "sln",
            "zl" :  zs_dict["zl"],
            "zs" : zs_dict["zs"],
            "ddt_params" : {"lambda_" : float(raw[2]),
                            "mu" : float(raw[3]),
                            "sigma" : float(raw[4])},
            "dd_params" : {"lambda_" : float(raw[5]),
                           "mu" : float(np.log(raw[6])),
                           "sigma" : float(raw[7])}}

def _load_lens_hist1d_weighted(filepath, zs_dict, ddt_max = 8000.0, bandwidth=20, nbins=400):
    _check_file(filepath)
    raw = np.genfromtxt(filepath, comments="#", encoding="utf-8", delimiter=",", names=True)
    ddt = raw["Dt"].astype(np.float64)
    weights = raw["weight"].astype(np.float64)

    mask = np.isfinite(ddt) & np.isfinite(weights) & (ddt < ddt_max)
    ddt = ddt[mask]
    weights = weights[mask]

    lens_name = os.path.splitext(os.path.basename(filepath))[0]
    hist1d = _build_kde1d(ddt, lens_name, weights=weights, bandwidth=bandwidth, nbins=nbins)

    return {"type" : "hist1d",
            "zl" :  zs_dict["zl"],
            "zs" : zs_dict["zs"],
            "hist1d" : hist1d}


def _load_lens_hist2d(filepath, zs_dict, col_dd, col_ddt, delimiter, bandwidth=20, nbins=200):

    _check_file(filepath)
    raw = np.genfromtxt(filepath, comments="#", encoding="utf-8", delimiter=delimiter)

    mask = np.all(np.isfinite(raw), axis=1)
    raw = raw[mask]
    
    ddt = raw[:, col_ddt].astype(np.float64)
    dd = raw[:, col_dd].astype(np.float64)
    lens_name = os.path.splitext(os.path.basename(filepath))[0]
    hist2d = _build_kde2d(dd, ddt, lens_name, bandwidth=bandwidth, nbins=nbins)

    return {"type" : "hist2d",
            "zl" :  zs_dict["zl"],
            "zs" : zs_dict["zs"],
            "hist2d" : hist2d}

def _load_lens_kde2d_full(filepath, zs_dict, bandwidth=20):
    _check_file(filepath)
    raw = np.genfromtxt(filepath, comments="#", encoding="utf-8", delimiter=",", skip_header=1)

    ddt = raw[:, 0].astype(np.float64)
    dd = raw[:, 1].astype(np.float64)
    lens_name = os.path.splitext(os.path.basename(filepath))[0]
    hist2d = _build_kde2d_full(dd, ddt, lens_name, bandwidth=bandwidth)

    return {"type" : "hist2d",
            "zl" :  zs_dict["zl"],
            "zs" : zs_dict["zs"],
            "hist2d" : hist2d}

def _validate(data, name, required):
    for key in required:
        if key not in data:
            raise ValueError(f"Falta clave '{key}' en dataset '{name}'")

    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)
                                            and v.ndim == 1}

    sizes = {k: v.size for k, v in arrays.items()}
    if len(set(sizes.values())) > 1:
      raise ValueError(f"[{name}] Arrays con tamaños inconsistentes: {sizes}")
    
    for key, arr in arrays.items():
        if np.any(~np.isfinite(arr)):
            n_bad = np.sum(~np.isfinite(arr))
            if name == "DESI": # En DESI esperamos algunos NaN, sólo advertimos
                print(f"[{name}] Advertencia: {n_bad} valores NaN o inf en '{key}'")
            else:
                raise ValueError(f"[{name}] Se encontraron {n_bad} valores NaN o inf en '{key}'")

      
    if "z" in data:
        if np.any(data["z"] <= 0):
            raise ValueError(f"[{name}] Hay {np.sum(data['z'] <= 0)} redshifts <= 0.")
    
    data_sigmas = {k : v for k, v in data.items() if "sigma" in k.lower()}
    for key, sigma in data_sigmas.items():
        if np.any(sigma <= 0):
            n_bad = np.sum(sigma <= 0)
            raise ValueError(f"[{name}] Se encontraron {n_bad} valores no positivos en '{key}'")

def _summarize_desi(data):
    print(f"\n{'─'*45}")
    print("Dataset: DESI BAO")
    print(f"{'─'*45}")
    for t in data["tracers"]:
        print(f"  Tracer {t['name']}: zeff={t['zeff']:.3f}, tipo={t['type']}, obs={t['obs']}, sigma={t['sigma']}")
    print(f"{'─'*45}\n")

def _summarize_bhs(data):
    print(f"\n{'─'*45}")
    print("Dataset: Black Hole Shadows")
    print(f"{'─'*45}")
    for i, name in enumerate(data["names"]):
        print(f" BH {name:8s}, z = {data['z'][i]:.5f}"
              f" R = {data['R'][i]:.3f} ± {data['dR'][i]:.3f}"
              f" M = {data['M'][i]:.3e} ± {data['dM'][i]:.3e}")
    print(f"{'─'*45}\n")

def summarize(data, name, n_cut=0):
    """
    Imprime un resumen del dataset cargado.
    Útil para verificar que la carga fue correcta.
    """
    z = data["z"]
    print(f"\n{'─'*45}")
    print(f"Dataset: {name}")
    print(f"{'─'*45}")
    print(f"  SNe/puntos       : {len(z)}")
    if n_cut > 0:
        print(f"  Eliminados (QC)  : {n_cut}")
    print(f"  z rango          : [{z.min():.4f}, {z.max():.4f}]")
 
    if name == "OHD":
        Hz = data["Hz"]
        print(f"  H(z) rango       : [{Hz.min():.1f}, {Hz.max():.1f}] km/s/Mpc")
 
    if name == "Pantheon":
        mb   = data["m_b_corr"]
        nc   = data["IS_CALIBRATOR"].sum()
        print(f"  m_b_corr rango   : [{mb.min():.2f}, {mb.max():.2f}]")
        print(f"  Calibradoras     : {nc}")
        print(f"  Hubble Flow      : {len(z) - nc}")
        diag = np.diag(data["cov"])
        print(f"  sigma_diag rango : [{np.sqrt(diag.min()):.4f}, "
              f"{np.sqrt(diag.max()):.4f}] mag")
 
    print(f"{'─'*45}\n")
