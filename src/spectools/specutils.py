
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
from spectools.spectrum_io import interpolate_spec
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import label
from ifscube.io import line_fit
from pathlib import Path

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


def convert_sdss_fits_to_txt(file):
    """
    Read an SDSS spectrum FITS file and prepare it for STARLIGHT input.

    The function extracts flux, wavelength, and errors from the SDSS file,
    corrects the spectrum to the rest frame using the most common redshift,
    and interpolates it with `interpolate_spec`.

    Parameters
    ----------
    file : str
        Path to the SDSS spectrum file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with rest-frame wavelength, flux, and flux error.
        Returns None if the filename does not match the expected format.
    """

    spec = fits.open(file)
    spec[1].data

    flux    = spec[1].data['flux']
    loglam  = spec[1].data['loglam']
    ivar    = spec[1].data['ivar']

    eflux = 1 / np.sqrt(ivar)
    wavelength = 10**loglam

    #Correcting waveleght to the rest frame.
    redshift = spec['SPZLINE'].data['LINEZ']
    redshift = redshift[redshift > 0]

    # z = round(stats.mode(redshift)[0], 5)
    moda = stats.mode(redshift)[0][0]
    redshift = round(float(moda), 5)

    rest_wave = wavelength / (1 + redshift)

    df = interpolate_spec(wave=rest_wave, flux=flux, eflux=eflux)
    return df


def create_starlight_input_file(data_directory='./../A194/', n_files: int = 500, 
                                library_directory= './CB19CSFBasesDir/', 
                                library= 'CB19_16x5',
                                IsErrSpecAvailable: bool = False, 
                                IsFlagSpecAvailable: bool = False, 
                                list_galaxies: list = []):

    library = 'CB19_16x5'
    mask = "mask_sdss.gm"
    if library == 'CB19_16x5':
        template = "CBASE.PARSEC.chab.16x5.all"
    elif library == 'CB19_16x12':
        template = "CBASE.cb19.PARSEC.chab.16x12.man.all"

    conf_file = f"""{n_files}                                        [Number of fits to run]
{library_directory}                                [base_dir]
{data_directory}                         [obs_dir]
./                          [mask_dir]
{data_directory}                         [out_dir]
-2007200                                         [your phone number]
5250.0                                           [llow_SN]   lower-lambda of S/N window
5260.0                                           [lupp_SN]   upper-lambda of S/N window
3569.0                                           [Olsyn_ini] lower-lambda for fit
9650.0                                           [Olsyn_fin] upper-lambda for fit
1.0                                              [Odlsyn]    delta-lambda for fit
1.0                                              [fscale_chi2] fudge-factor for chi2
FIT                                              [FIT/FXK] Fit or Fix kinematics
{1 if IsErrSpecAvailable else 0}                                                [IsErrSpecAvailable]  1/0 = Yes/No
{1 if IsFlagSpecAvailable else 0}                                                [IsFlagSpecAvailable] 1/0 = Yes/No
"""

    # Componentes fixos
    fixos = [f"StCv04.C11.config", template, 
             f'{mask}', "CCM", "0.0", "150.0"]
    

    for file in list_galaxies:
        file_name = file.split('.')[0] # nome do arquivo sem extensão
        out_name = f"{file_name}_{library}"
        conf_file += f"{file}  {'  '.join(fixos)}  {out_name}\n"
        # print(conf_file)
        # conf_file += "\n"

    return conf_file


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


def read_starlight_output(files: list[str], filepath: str) -> dict:
    """
    Parse and extract information from STARLIGHT output files.

    This function reads one or more STARLIGHT output files, extracting both
    metadata and synthetic spectrum information. For each galaxy (identified
    by the prefix of the filename), the function stores:
    
    - Metadata key-value pairs parsed from lines in the format `value [description]`.
    - Synthetic spectrum data, including wavelength, observed flux, model flux,
      and weights.

    Parameters
    ----------
    files : list of str
        List of STARLIGHT output filenames to be read.
        Each filename is expected to start with the galaxy name
        (e.g., ``NGC1234_output.txt`` → galaxy name = ``NGC1234``).
    filepath : str
        Path to the directory containing the STARLIGHT output files.

    Returns
    -------
    dict
        A nested dictionary with the following structure:
        
        {
            galaxy_name: {
                'metadata': dict
                    Key-value pairs extracted from the header section.
                'wavelenght': list of float
                    Wavelength values of the synthetic spectrum.
                'f_obs': list of float
                    Observed flux values.
                'f_model': list of float
                    Model flux values.
                'weight': list of float
                    Weights associated with each wavelength point.
            },
            ...
        }
    """

    results_all = {}
    reading_spectrum = False

    for file in files:
        galaxy_name = file.split('_')[0]

        if galaxy_name not in results_all:
            results_all[galaxy_name] = {}
    
        results_all[galaxy_name] = {
            'metadata': {},
            'wavelenght': [],
            'f_obs': [],
            'f_model': [],
            'weight': [],
            'residual': []
        }

        with open(filepath+file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            for idx, line in enumerate(lines):
                line = line.strip()

                match = re.match(r'^(.*?)\[(.*)\]', line)
                if match:
                    raw_values, description = match.groups()
                    raw_values = raw_values.strip()
                    description = description.strip()
            
                    values = raw_values.split()
                    if len(values) == 1:
                        value = values[0]  # valor único como string
                    else:
                        value = values  # lista de strings
            
                    results_all[galaxy_name]['metadata'][description] = value

                elif line.startswith('## Synthetic spectrum '):  # Início dos espectros # elif line:  # Início dos espectros
                    reading_spectrum = True
                    continue
                
                elif reading_spectrum:
                    parts = line.split()
                    if len(parts) == 4:
                        spectrum = list(map(float, parts))
                        results_all[galaxy_name]['wavelenght'].append(spectrum[0])
                        results_all[galaxy_name]['f_obs'].append(spectrum[1])
                        results_all[galaxy_name]['f_model'].append(spectrum[2])
                        results_all[galaxy_name]['weight'].append(spectrum[3])
                    else:
                        reading_spectrum = False  # Parar de ler se a linha mudar o formato
                    continue  # Já tratou essa linha, então pula pro próximo loop

    return results_all


def create_new_fits_from_starlight(data):
    """
    Create a FITS file from STARLIGHT output data.

    Builds a multi-extension FITS with observed spectrum, stellar model,
    and gas component (observed - stellar). Metadata is used to define
    the wavelength axis in the header.

    Parameters
    ----------
    data : dict
        STARLIGHT output with keys 'f_obs', 'f_model', and 'metadata'.

    Returns
    -------
    astropy.io.fits.HDUList
        FITS object with OBSERVED, STELLAR, and GAS extensions.
    """

    observed    = data['f_obs']
    stellar     = data['f_model']
    gas         = np.array(observed) - np.array(stellar)

    hdu = fits.Header()
    crval = float(data['metadata']['l_ini (A)'])
    cdelt = float(data['metadata']['dl    (A)'])
    hdu['NAXIS'] = 1
    hdu['CRPIX1'] = 1 
    hdu['CRVAL1'] = crval #/ (1 + redshift)
    hdu['CDELT1'] = cdelt
    hdu['CTYPE1'] = 'WAVE'
    hdu['CUNIT1'] = 'Angstrom'

    # Base extensions
    extensions = [
        (observed, "OBSERVED"),
        (stellar, "STELLAR"),
        (gas, "GAS"),
    ]

    # Build HDUList
    hdus = [fits.ImageHDU(data=dataa, header=hdu, name=name) 
            for dataa, name in extensions]

    return fits.HDUList([fits.PrimaryHDU(), *hdus])


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
