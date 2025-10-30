
import os
import json
import glob
# import wget
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from astropy import units as u
from SciServer import SkyServer
import SciServer
from astropy.coordinates import SkyCoord


def search_skyserver(galaxy, SkyServer_DataRelease):
    """
    Query the SDSS SkyServer for galaxies matching the input sample.

    Parameters
    ----------
    galaxy : pandas.DataFrame
        DataFrame containing the target galaxies. Must include the columns
        "ra", "dec", and "z".
    SkyServer_DataRelease : str
        Identifier of the SDSS data release to query (e.g., "DR18").

    Returns
    -------
    pandas.DataFrame
        DataFrame with the matching spectroscopic objects retrieved from
        the SkyServer.

    Notes
    -----
    The function constructs and executes an SQL query on the SDSS SkyServer.
    It uses the SpecObj table and applies a WHERE clause to filter results
    by right ascension (RA), declination (DEC), and spectroscopic redshift (z).
    """

    # Get the minimum and maximum RA, DEC and zspec of the given galaxy
    minra   =min(galaxy['ra'])-0.0001
    maxra   =max(galaxy['ra'])+0.001
    mindec  =min(galaxy['dec'])-0.0001
    maxdec  =max(galaxy['dec'])+0.001
    minz    =min(galaxy['z'])-0.0001
    maxz    =max(galaxy['z'])+0.001

    # Build the SQL query
    SkyServer_TestQuery = "SELECT ra, dec, z, specobjid, class, plate, mjd, fiberid, run2d, bestObjID " \
                        "from SpecObj " \
                        f"WHERE class = 'galaxy' AND ra BETWEEN {minra} AND {maxra} AND dec BETWEEN {mindec} AND {maxdec} AND z BETWEEN {minz} AND {maxz} " \

    # Execute the query
    query = SkyServer.sqlSearch(sql=SkyServer_TestQuery, dataRelease=SkyServer_DataRelease)

    return query


def match_galaxies(survey, sdss):
    """
    Matches the galaxies between one survey X and SDSS datasets.

    Parameters
    ----------
    survey : pandas.DataFrame
        DataFrame containing the survey X galaxies.
    sdss : pandas.DataFrame
        DataFrame containing the SDSS galaxies.

    Returns
    -------
    df_common_filtered : pandas.DataFrame
        DataFrame with the common galaxies between SPLUS and SDSS.
    """

    # Convert the RA and DEC columns to astropy SkyCoord objects
    coords_survey    = SkyCoord(ra=survey["ra"].values * u.degree, dec=survey["dec"].values * u.degree)
    coords_sdss     = SkyCoord(ra=sdss["ra"].values * u.degree, dec=sdss["dec"].values * u.degree)

    # Tolerance for the match
    tolerance = 1 * u.arcsec

    # Match the galaxies between the two datasets
    idx, d2d, _ = coords_sdss.match_to_catalog_sky(coords_survey)

    # Create a mask for the matches
    mask = d2d < tolerance

    # Get the matched DataFrame
    df_matched = survey.iloc[idx].reset_index(drop=True)

    # Concatenate the two DataFrames
    df_common = pd.concat([sdss.reset_index(drop=True), 
                        df_matched.drop(columns=["ra", "dec", "z"])],
                        axis=1)

    # Filter the DataFrame with the mask
    df_common_filtered = df_common[mask]

    return df_common_filtered

