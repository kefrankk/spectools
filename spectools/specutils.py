
import wget
import os
import re
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import label
from ifscube.io import line_fit
from pathlib import Path
from sparcl.client import SparclClient
from SciServer import SkyServer



# Define your own figure configuration here
font = 12
plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # ou Helvetica se estiver instalada
    'text.usetex': True,
    'font.size': font,
    'axes.titlesize': font,
    'axes.labelsize': font,
    'xtick.labelsize': font,
    'ytick.labelsize': font,
    'legend.fontsize': font,
    'figure.titlesize': font,
    'font.sans-serif': ['Helvetica']
})


def get_desi_spec(cons):
    """
    Queries the SPARCL database and returns a pandas DataFrame containing the results of the query.

    Parameters
    ----------
    cons : dict
        A dictionary containing the constraints to be used in the query.

    Returns
    -------
    results_desi_dr1 : pandas.DataFrame
        A pandas DataFrame containing the results of the query.
    """
    
    print(f'Initiating the download of DESI data...')
    client = SparclClient()
    
    out = ['sparcl_id', 'ra', 'dec', 'redshift', 'spectype', 'objtype', 'data_release', 'desiname', 'zcat_nspec', 'targetid']

    print('Running client.find...')
    found_I = client.find(outfields=out, constraints=cons, limit=None)
    
    # Define the fields to include in the retrieve function
    inc = ['sparcl_id', 'specid', 'targetid', 'data_release', 'redshift', 'flux',
           'wavelength', 'model', 'ivar', 'mask', 'wave_sigma', 'spectype', 'ra', 'dec']
    
    ids_I = found_I.ids
    print('Running client.retrieve...')
    results_I = client.retrieve(uuid_list=ids_I, include=inc, limit=None, dataset_list = ['DESI-DR1'])
    
    results_desi_dr1 = pd.json_normalize(results_I.records)

    return results_desi_dr1


def search_skyserver(galaxy, SkyServer_DataRelease: str = "DR18"):
    """
    Searches the SkyServer database for galaxies that match the RA, DEC and zspec of the given galaxy.      

    Parameters
    ----------
    galaxy : pandas.DataFrame
        DataFrame containing the galaxies to search for.
    data_release : str
        The SDSS data release to use for the search.

    Returns
    -------
    query : pandas.DataFrame
        DataFrame containing the results of the search.     

    Notes
    -----
    The function builds a SQL query with the given parameters and executes it on the SkyServer database.
    The query is built using the SpecObj table of the SDSS and the WHERE clause is used to filter the results
    by the RA, DEC and zspec of the given galaxy.
    """

    # Get the minimum and maximum RA, DEC and zspec of the given galaxy
    minra   =min(galaxy['ra'])#-0.01
    maxra   =max(galaxy['ra'])#+0.01
    mindec  =min(galaxy['dec'])#-0.01
    maxdec  =max(galaxy['dec'])#+0.01
    minz    =min(galaxy['z'])#-0.01
    maxz    =max(galaxy['z'])#+0.01

    # Build the SQL query
    SkyServer_TestQuery = "SELECT ra, dec, z, specobjid, class, plate, mjd, fiberid, run2d, bestObjID " \
                        "from SpecObj " \
                        f"WHERE class = 'galaxy' AND ra BETWEEN {minra} AND {maxra} AND dec BETWEEN {mindec} AND {maxdec} AND z BETWEEN {minz} AND {maxz} and zWarning = 0" \

    # Execute the query
    query = SkyServer.sqlSearch(sql=SkyServer_TestQuery, dataRelease=SkyServer_DataRelease)

    return query


def download_sdss_spectra(sdss_data, URL, download_dir):
    """
    Downloads SDSS spectra based on the provided DataFrame containing spectrum information.

    Parameters
    ----------
    sdss_data : pandas.DataFrame
        DataFrame containing the SDSS spectrum information, including 'run2d', 'plate', 'mjd', and 'fiberid'.
    URL : str
        Base URL for downloading the spectra.
    download_dir : str
        Directory where the spectra will be downloaded. 

    Returns
    -------
    None
    """

    for _, row in sdss_data.iterrows():
        run2d = row['run2d']
        plate = row['plate']
        fiberid = row['fiberid']
        mjd = row['mjd']

        spec = f'spec-{str(plate).zfill(4)}-{mjd}-{str(fiberid).zfill(4)}.fits'

        url = f'{URL}/{run2d}/spectra/lite/{str(plate).zfill(4)}/{spec}'
        if not url:
            print('No URL found for the spectrum.')

        if spec not in os.listdir(download_dir):
            wget.download(url, out = download_dir)



def match_galaxies(splus, survey):
    """
    Matches galaxies between the SPLUS dataset and another survey.
    
    Parameters
    ----------
    splus : pandas.DataFrame
        DataFrame containing galaxies from the SPLUS survey.
    survey : pandas.DataFrame
        DataFrame containing galaxies from the comparison survey.
    
    Returns
    -------
    df_common_filtered : pandas.DataFrame
        DataFrame containing galaxies found in both SPLUS and the comparison survey.
    """

    # Convert the RA and DEC columns to astropy SkyCoord objects
    coords_splus    = SkyCoord(ra=splus["ra"].values * u.degree, dec=splus["dec"].values * u.degree)

    coords     = SkyCoord(ra=survey["ra"].values * u.degree, dec=survey["dec"].values * u.degree)
    
    # Tolerance for the match
    tolerance = 0.5 * u.arcsec

    # Match the galaxies between the two datasets
    idx, d2d, _ = coords.match_to_catalog_sky(coords_splus)

    # Create a mask for the matches
    mask = d2d < tolerance

    # Get the matched DataFrame
    df_matched = splus.iloc[idx].reset_index(drop=True)

    # Concatenate the two DataFrames
    df_common = pd.concat([survey.reset_index(drop=True), df_matched], axis=1)

    # Filter the DataFrame with the mask
    df_common_filtered = df_common[mask]

    return df_common_filtered


def run_starlight(starlight_dir):
    """
    Run the STARLIGHT software with the given configuration file.

    Parameters
    ----------
    conf_file : str
        Path to the STARLIGHT configuration file.
    """
    # subprocess.run(f'cd {os.path.dirname(starlight_dir)}', shell=True)
    # subprocess.run('pwd', shell=True    )
    directory = Path(starlight_dir)
    subprocess.run(f"./StarlightChains_v04.exe < config.in ", shell=True, cwd=directory)


def select_do_fit(galaxy, value:int = 3, hb_detected: bool = False):
    """
    Decide whether a galaxy spectrum should be fitted.

    Opens the FITS file, measures noise and line strengths in predefined
    wavelength windows, and checks if emission lines (Hα, and optionally Hβ)
    are strong enough compared to the local noise.

    Parameters
    ----------
    galaxy : str
        Path to the galaxy FITS file.
    value : int, optional
        Detection threshold multiplier for line significance (default is 3).
    hb_detected : bool, optional
        If True, requires both Hβ and Hα detection; otherwise only Hα.

    Returns
    -------
    bool
        True if the spectrum meets the detection criteria, False otherwise.
    """

    a = fits.open(galaxy)
    
    header = a["OBSERVED"].header
    gas = a['OBSERVED'].data

    bb_window   = [3600, 3700]
    oii_window  = [3700, 3760]

    blue_window = [4600, 4700]
    oiii_window = [4920, 5100]
    hb_window   = [4800, 4920]

    red_window  = [6100, 6200]
    ha_window   = [6450, 6650]
    sii_window  = [6700, 6780]

    lam     = (np.arange(0, gas.shape[0]) * header['cdelt1'] + header['crval1'])

    std_bb      = np.std(gas[(lam >= bb_window[0]) & (lam <= bb_window[1])])
    std_blue    = np.std(gas[(lam >= blue_window[0]) & (lam <= blue_window[1])])
    std_red     = np.std(gas[(lam >= red_window[0]) & (lam <= red_window[1])])

    oii_max     = np.max(gas[(lam >= oii_window[0]) & (lam <= oii_window[1])])
    hb_max      = np.max(gas[(lam >= hb_window[0]) & (lam <= hb_window[1])])
    oiii_max    = np.max(gas[(lam >= oiii_window[0]) & (lam <= oiii_window[1])])
    ha_max      = np.max(gas[(lam >= ha_window[0]) & (lam <= ha_window[1])])
    sii_max     = np.max(gas[(lam >= sii_window[0]) & (lam <= sii_window[1])])

    if hb_detected:
        if hb_max > value* std_blue and ha_max > value* std_red:
            return True
        else:
            return False
    else:
        if ha_max > value* std_red:
            return True
        else:
            return False
        

def sort_by_pattern(file):
    name = file.stem  # sem extensão
    
    # Padrão SDSS: spec-PLATE-MJD-FIBERID
    match_sdss = re.match(r"spec-(\d+)-(\d+)-(\d+)", name)
    if match_sdss:
        plate, mjd, fiberid = map(int, match_sdss.groups())
        return (0, plate, mjd, fiberid)  # "0" força SDSS a vir antes
    
    # Padrão UUID-like
    match_uuid = re.match(r"spec-[0-9a-f]{8}-", name)
    if match_uuid:
        return (1, name)  # "1" garante que UUIDs fiquem depois
    
    # fallback: string normal
    return (2, name)


def interpolate_spec(wave, flux, eflux):
    """
    Interpolates a spectrum using linear interpolation.

    Parameters
    ----------
    wave : array
        The wavelengths of the spectrum.
    flux : array
        The fluxes of the spectrum.
    eflux : array
        The errors of the fluxes of the spectrum.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the interpolated spectrum.

    Notes
    -----
    The errors are also interpolated and filled with the minimum error
    value in case of non-finite values.
    """

    #Cria novo eixo de comprimento de onda com passo de 1 Å
    wavelength_linear = np.arange(np.ceil(wave.min()), np.floor(wave.max()) + 1, 1)

    # Interpola o fluxo
    interp_flux = interp1d(wave, flux, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_error = interp1d(wave, eflux, kind='linear', bounds_error=False, fill_value="extrapolate")

    flux_linear = interp_flux(wavelength_linear)
    error_linear = interp_error(wavelength_linear)

    error_linear = np.where(~np.isfinite(error_linear), min(eflux), error_linear)

    df = pd.DataFrame({'wavelength': wavelength_linear, 'flux': flux_linear, 'flux_error': error_linear}) #, 'flux_error': flux_error})

    return df


def make_line(rest_wavelength, velocity=None, sigma=None, amplitude="peak, 0:", k_group=None):
    line = {
        "rest_wavelength": rest_wavelength,
        "amplitude": amplitude,
    }
    if velocity is not None:
        line["velocity"] = velocity
    if sigma is not None:
        line["sigma"] = sigma
    if k_group is not None:
        line["k_group"] = k_group
    return line



def plot_starlight_spectrum(data):
    metadata    = data['metadata']
    lam         = np.array(data['wavelenght'])
    f_obs       = np.array(data['f_obs'])
    f_model     = np.array(data['f_model'])
    gas         = np.array(np.array(f_obs) - np.array(f_model))
    weight     = np.array(data['weight'])

    limits_list = [(lam > 3750) & (lam < 4050), (lam > 4200) & (lam < 4500), (lam > 6450) & (lam < 6750)]

    # Identify emission line intervals
    w0      = np.array(weight) <= 0
    labels, num_features = label(w0)

    intervals = []
    for i in range(1, num_features + 1):
        indices = np.where(labels == i)[0]
        inicio = lam[indices[0]]
        fim = lam[indices[-1]]
        intervals.append((inicio, fim))


    fig = plt.figure(figsize=(14, 8))

    gd = mpl.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1])
    gs_top = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gd[0], height_ratios=[3, 1], hspace=0)

    ax_top = fig.add_subplot(gs_top[0, :])
    ax_bottom = fig.add_subplot(gs_top[1, :], sharex=ax_top)

    # Principal plot
    ax_top.plot(lam, f_obs, label='Observed', lw=1, color='k')
    ax_top.plot(lam, f_model, label='Model', lw=2, color='r')
    ax_top.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    for  i, (inicio, fim) in enumerate(intervals):
        labels = 'Masked region' if i == 0 else None
        ax_top.axvspan(inicio, fim, color='orange', alpha=0.3, label=labels)
    ax_top.legend()

    # Metadados
    ax_top.text(0.85, 0.2, f'Adev: {metadata["adev (%)"]}', transform=ax_top.transAxes, ha='left')
    ax_top.text(0.85, 0.15, f'Chi2: {metadata["chi2/Nl_eff"]}', transform=ax_top.transAxes, ha='left')
    ax_top.text(0.85, 0.1, f'Reddening: {metadata["AV_min  (mag)"]}', transform=ax_top.transAxes, ha='left')
    ax_top.set_ylabel(r'flux (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA$)')

    # Observed - stellar
    ax_bottom.plot(lam, gas, lw=1, color='gray')
    ax_bottom.set_ylabel('residue')
    ax_bottom.set_ylim(-0.3, 0.3)
    ax_bottom.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax_bottom.set_xlabel(r'rest wavelength ($\AA$)')


    gs_bot = mpl.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gd[1])

    for i, limit in enumerate(limits_list):

        ax = fig.add_subplot(gs_bot[i])
        ax.plot(lam[limit], f_obs[limit], label='Observed', lw=1, color='k' )
        ax.plot(lam[limit], f_model[limit], label='Model', lw=1, color='r')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlabel(r'rest wavelength ($\AA$)')
        ax.set_ylim(0, 2)

        if i == 0:
            # ax.legend()
            ax.set_ylabel(r'flux (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA$)')


    fig.tight_layout()

    return fig


def plot_ifscube_components(fit, ax, ax_lim, add_label = False):
    """
    Plot individual spectral components from an IFSCube fit.

    For each feature in the fit, computes its contribution and plots it
    on the current axis over the given wavelength range.

    Parameters
    ----------
    fit : object
        Fit result containing parameters, wavelengths, and functions.
    ax_lim : array-like of bool
        Boolean mask selecting the wavelength region to plot.
    add_label : bool, optional
        If True, adds a legend label to the first component.
    """

    ppf = fit.parameters_per_feature
    for i in range(0, len(fit.parameter_names), ppf):
        feature_wl = fit.feature_wavelengths[int(i / ppf)]
        parameters = np.array(fit.solution[i:i + ppf], dtype=np.float64)
        feature_wl_arr = np.array([feature_wl], dtype=np.float64)
        line = fit.function(fit.wavelength, feature_wl_arr, parameters)
        label = 'Components' if add_label and i == 0 else None
        ax.plot(fit.wavelength[ax_lim], fit.pseudo_continuum[ax_lim] + line[ax_lim], '--', color='#20c073', label=label)


def plot_ifscube_spectra(directory, galaxy):
    """
    Plot observed, model, and continuum spectra for a galaxy.

    Loads the spectral fit from the given directory, selects six wavelength
    regions, and creates a 2x3 grid of plots showing observed data,
    pseudo-continuum, model, and fitted components.

    Parameters
    ----------
    directory : str
        Path to the folder with the fit results.
    galaxy : str
        Galaxy identifier or filename.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the spectral plots.
    """
    
    fit                 = line_fit.load_fit(directory/galaxy)
    feature_names       = fit.feature_names
    wave                = fit.wavelength
    obs                 = fit.data
    stellar             = fit.stellar
    solution            = np.ascontiguousarray(fit.solution, dtype=np.float64)
    model               = fit.function(wave, fit.feature_wavelengths, solution)      
    continuum           = fit.pseudo_continuum
        
    galaxy_name = galaxy.split('_')[0]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    fig.suptitle(f'Spectra for {galaxy_name}', fontsize=14)
    colors = ['#20c073', '#073be5']

    limits_list = [(wave > 3650) & (wave < 3800), (wave > 4290) & (wave < 4440), (wave > 4790) & (wave < 4920), 
                   (wave > 4910) & (wave < 5050), (wave > 6490) & (wave < 6630), (wave > 6650) & (wave < 6790)]

    for i, limit in enumerate(limits_list):

        ax = axs.flat[i]
        ax.plot(wave[limit], obs[limit], label='Observed', lw=1, color='k' )
        ax.plot(wave[limit], continuum[limit], label='Pseudo continuum', lw=1, color='C1')
        plot_ifscube_components(fit, ax, limit, add_label=(i == 0))
        ax.plot(wave[limit], model[limit], label='Model', lw=1, color=colors[1])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlabel(r'rest wavelength ($\AA$)')
        ax.set_ylabel(r'flux (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA$)')
    
        if i == 0:
            ax.legend()
    
    fig.tight_layout()
    return fig


def emission_lines():
    return {
    "Hb_4861": 4861.325,
    "Ha_6563": 6562.8,
    "OII_3726": 3726.032,
    "OII_3729": 3728.815,
    "OIII_4363": 4363.210,
    "NII_6583": 6583.46,
    "NII_6548": 6548.04,
    "OIII_5007": 5006.84,
    "OIII_4959": 4958.91,
    "SII_6716": 6716.44,
    "SII_6731": 6730.82,
    }