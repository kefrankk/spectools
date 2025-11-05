
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from ifscube.io import line_fit
from spectools.specutils import interpolate_spec
from scipy import stats



def to_txt_desi(data: pd.DataFrame, saving_dir: str):
    """
    Writes a DataFrame to a text file.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be written to a text file.
    saving_dir : str
        The directory where the text file will be saved.

    Notes
    -----
    The text file will be written to a directory specified by the user.
    The filename will be the SPARCL ID of the object.
    The file will contain the restframe wavelength, flux and flux error.
    The file will be written in the format: wavelength flux flux_error.
    """
    for obj, row in data.iterrows():
        flux        = row['flux']
        ivar        = row['ivar']
        wavelength  = row['wavelength']
        redshift    = row['redshift']

        sparclid    = row['sparcl_id']

        eflux       = 1 / np.sqrt(ivar)
        rest_wave = wavelength / (1 + redshift)

        df = interpolate_spec(wave=rest_wave, flux=flux, eflux=eflux)

        df.to_csv(f'{saving_dir}spec-{sparclid}.txt', sep=' ', header=False, index=False)


def to_fits_desi(data: pd.DataFrame):
    """
    Convert spectra stored in a DataFrame into FITS (Flexible Image Transport System) files.

    For each row in the input DataFrame, the function:
    - Computes the rest-frame wavelength (corrected for redshift).
    - Estimates the flux error from the inverse variance (`ivar`).
    - Interpolates the spectrum onto a uniform wavelength grid.
    - Creates a multi-extension FITS (MEF) file containing:
        * PRIMARY: header with WCS (World Coordinate System) information.
        * FLUX: interpolated spectrum (flux).
        * IVAR: flux uncertainties.

    The file is saved in the specified directory, using the `sparcl_id` as the filename.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the spectra. Must include the following columns:
        - 'wavelength' : array-like
            Observed wavelengths.
        - 'flux' : array-like
            Flux values corresponding to each wavelength.
        - 'ivar' : array-like
            Inverse variance of the flux.
        - 'redshift' : float
            Redshift associated with the spectrum.
        - 'sparcl_id' : str or int
            Unique identifier used to name the FITS file.
    saving_dir : str
        Path to the directory where the FITS files will be saved.

    Returns
    -------
    None
        The function does not return anything. FITS files are written to disk.

    Notes
    -----
    - The spectrum is interpolated using the helper function `interpolate_spec`.
    - The WCS is defined in one dimension (wavelength).
    - Existing files with the same name will be overwritten.
    """


    for obj, row in data.iterrows():
        wavelength  = row['wavelength']
        flux        = row['flux']
        ivar        = row['ivar']
        redshift    = row['redshift']

        sparclid    = row['sparcl_id']

        eflux       = 1 / np.sqrt(ivar)
        rest_wave = wavelength / (1 + redshift)

        cdelt = 1
        crpix = 1
        crval = rest_wave[0]

        wcs = WCS(naxis=1)
        wcs.wcs.crpix  = [crpix]
        wcs.wcs.cdelt  = [cdelt]
        wcs.wcs.crval  = [crval]
        wcs.wcs.ctype  = ['WAVE']

        df = interpolate_spec(wave=rest_wave, flux=flux, eflux=eflux)

        header = wcs.to_header()

        # Creates the MEF file
        h = fits.HDUList()
        hdu = fits.PrimaryHDU(header=header)
        hdu.name = 'PRIMARY'
        h.append(hdu)

        # Creates the observed spectrum extension
        hdu = fits.ImageHDU(data=df.flux, header=header)
        hdu.name = 'FLUX'
        h.append(hdu)

        # Creates the ivar extension.
        hdu = fits.ImageHDU(data=df.flux_error, header=header)
        hdu.name = 'IVAR'
        h.append(hdu)

        return h


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


def generate_ifscube_configuration_file(
    redshift: float = 0.0, 
    continuum_degree: int = 1,
    # output_file: str = "ifscube_config.cfg",
    lines: dict = None, 
    function: str = "gaussian", 
    scidata: str = "GAS",
    # scidata: str = "OBSERVED",
    stellar: bool = False,
    out_image: str = None
    ):
    
    suffix = "_g" if function == "gaussian" else "_gh"

    # Cabeçalho fixo
    config = f"""[fit]
fitting_window: 3630:7000
fit_continuum: yes 
function: {function}
guess_parameters: yes
optimization_window: 7
optimize_fit: yes
overwrite: yes
# suffix: {suffix}
out_image: {out_image}
test_jacobian: no
trivial: no
verbose: no
write_fits: yes
# method: slsqp
# monte_carlo: 100

[loading]
primary: 0
scidata: {scidata}
{'stellar: STELLAR' if stellar else '# stellar: STELLAR'}
# redshift: {redshift}

[minimization]
eps: 1e-2
ftol: 1e-5
disp: no
maxiter: 1000

[continuum]
degree: {continuum_degree}
n_iterate: 10
lower_threshold: 1
upper_threshold: 1
[equivalent_width]
sigma_factor: 6
    """

    # Adiciona as linhas espectrais definidas
    for line_name, params in lines.items():
        if not params.get("include", True):
            continue  # Pula se não for incluir

        config += f"\n[{line_name}]\n"
        config += f"rest_wavelength: {params['rest_wavelength']}\n"
        if "velocity" in params:
            config += f"velocity: {params['velocity']}\n"
        if "sigma" in params:
            config += f"sigma: {params['sigma']}\n"
        if "amplitude" in params:
            config += f"amplitude: {params['amplitude']}\n"
        if "k_group" in params:
            config += f"k_group: {params['k_group']}\n"

    return config


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


def read_ifscube_output(directory, galaxy, ifscube_output):
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

    ifscube_output[galaxy_name] = {}

    continuum = np.nan_to_num(continuum, nan=0.0)


    ax02_lim = (wave > 3760) & (wave < 3810)
    ax03_lim = (wave > 4750) & (wave < 4800)
    ax04_lim = (wave > 6500) & (wave < 6780)
    ax05_lim = (wave > 4500) & (wave < 4600)
    std_oii   = np.std(obs[ax02_lim])
    std_hb    = np.std(obs[ax03_lim])
    std_ha    = np.std(obs[ax04_lim])
    std_region = obs[ax05_lim]

    ifscube_output[galaxy_name]["line_model"] = {}
    ifscube_output[galaxy_name]['flux'] = {}
    ifscube_output[galaxy_name]['amplitude'] = {}
    ifscube_output[galaxy_name]['velocity'] = {}
    ifscube_output[galaxy_name]['sigma'] = {}
    ifscube_output[galaxy_name]['std_oii'] = std_oii
    ifscube_output[galaxy_name]['std_hb'] = std_hb
    ifscube_output[galaxy_name]['std_ha'] = std_ha
    ifscube_output[galaxy_name]['std_region'] = std_region.tolist()  # save one region of the spectra to define the threshold for the emission lines cut

    ifscube_output[galaxy_name]['wavelength'] = wave.tolist()
    ifscube_output[galaxy_name]['pseudo_continuum'] = continuum.tolist()
    ifscube_output[galaxy_name]['model'] = model.tolist()

    ppf = fit.parameters_per_feature
    for i in range(0, len(fit.parameter_names), ppf):
        feature_name = fit.feature_names[int(i // ppf)] 
        feature_wl = fit.feature_wavelengths[int(i / ppf)]
        parameters = np.array(fit.solution[i:i + ppf], dtype=np.float64)
        feature_wl_arr = np.array([feature_wl], dtype=np.float64)
        line = fit.function(fit.wavelength, feature_wl_arr, parameters)
        ifscube_output[galaxy_name]['line_model'][feature_name] = line.tolist()


    for line in feature_names:

        ifscube_output[galaxy_name]['flux'][line] = fit.flux_model[feature_names.index(line)]
        ifscube_output[galaxy_name]['amplitude'][line] = fit.solution[feature_names.index(line)*3]
        ifscube_output[galaxy_name]['velocity'][line] = fit.solution[feature_names.index(line)*3 + 1]
        ifscube_output[galaxy_name]['sigma'][line] = fit.solution[feature_names.index(line)*3 + 2]
    
    return ifscube_output


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

