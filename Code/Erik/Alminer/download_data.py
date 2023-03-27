"""Modified version of alminer.download_data. Here it is possible to download n fits files.
   Default is to search through the categories specified and then randomly select n files 
   from all the observations matching that category, but we can also choose n random 
   observations. Look for lines with many ###### for info on how to change. """


##############################################
# Libraries
##############################################
from constants_copy import band_names, band_color, band_min_freq, band_max_freq, \
    CO_line_names, CO_line_freq, CO_line_ha, CO_line_label, VALID_KEYWORDS_STR, \
    NEW_COLUMNS, COLUMN_TYPES
from pyvo.dal import tap
from astroquery.alma import Alma
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import name_resolve
from astropy.coordinates import get_icrs_coordinates
from astropy.coordinates import Angle
import os
import re
import pandas as pd
import numpy as np
# import click to make queries like [Y/n]
import random

np.set_printoptions(threshold=np.inf)


def _format_bytes(size):
    """Convert the size of the dota to be downloaded in human-readable format."""
    power = 1000
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB',
                    3: 'GB', 4: 'TB', 5: 'PB', 6: 'EB'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]


def download_n_data(observations, amount_of_random_files, fitsonly=False, dryrun=False, print_urls=False, filename_must_include='',
                    location='./data', archive_mirror='ESO'):
    """
    Download ALMA data from the archive to a location on the local machine.

    Parameters
    ----------
    observations : pandas.DataFrame
         This is likely the output of e.g. 'conesearch', 'target', 'catalog', & 'keysearch' functions.
    amount_of_random_files : int
         Declare here how many random files you wish to download from the current search, if there are
         less downloadable files it will tell you how many there are and ask if you wish to continue.
    fitsonly : bool, optional
         (Default value = False)
         Download individual fits files only (fitsonly=True). This option will not download the raw data
         (e.g. 'asdm' files), weblogs, or README files.
    dryrun : bool, optional
         (Default value = False)
         Allow the user to do a test run to check the size and number of files to download without actually
         downloading the data (dryrun=True). To download the data, set dryrun=False.
    print_urls : bool, optional
         (Default value = False)
         Write the list of urls to be downloaded from the archive to the terminal.
    filename_must_include : list of str, optional
         (Default value = '')
         A list of strings the user wants to be contained in the url filename. This is useful to restrict the
         download further, for example, to data that have been primary beam corrected ('.pbcor') or that have
         the science target or calibrators (by including their names). The choice is largely dependent on the
         cycle and type of reduction that was performed and data products that exist on the archive as a result.
         In most recent cycles, the science target can be filtered out with the flag '_sci' or its ALMA target name.
    location : str, optional
         (Default value = ./data)
         directory where the downloaded data should be placed.
    archive_mirror : str, optional
         (Default value = 'ESO')
         The archive service to use. Options are:
         'ESO' for Europe (https://almascience.eso.org),
         'NRAO' for North America (https://almascience.nrao.edu), or
         'NAOJ' for East Asia (https://almascience.nao.ac.jp)

    """
    print("================================")
    # we use astroquery to download data
    myAlma = Alma()
    default_location = './data'
    myAlma.cache_location = default_location
    if archive_mirror == 'NRAO':
        mirror = "https://almascience.nrao.edu"
    elif archive_mirror == 'NAOJ':
        mirror = "https://almascience.nao.ac.jp"
    else:
        mirror = "https://almascience.eso.org"
    myAlma.archive_url = mirror
    # catch the case where the DataFrame is empty.
    try:
        if any(observations['data_rights'] == 'Proprietary'):
            print("Warning: some of the data you are trying to download are still in the proprietary period and are "
                  "not publicly available yet.")
            observations = observations[observations['data_rights'] == 'Public']
        uids_list = observations['member_ous_uid'].unique()
        ############################### Extract random samples from uids_list if you want to have different observations ###############################
        ###############################                       Remember to change below.                                  ###############################
        ###############################                Considerably faster but with less variance                        ###############################
        # NOTE There might be a different amount of fits files from observations
        # when len(uids_list) == 0, it's because the DataFrame included only proprietary data and we removed them in
        # the above if statement, so the DataFrame is now empty
        if len(uids_list) == 0:
            print("No data to download. Check the input DataFrame. It is likely that your query results include only "
                  "proprietary data which cannot be freely downloaded.")
            return
    # this is the case where the query had no results to begin with.
    except TypeError:
        print("No data to download. Check the input DataFrame.")
        return
    # change download location if specified by user, else the location will be a folder called 'data'
    # in the current working directory
    if location != default_location:
        if os.path.isdir(location):
            myAlma.cache_location = location
        else:
            print("{} is not a directory. The download location will be set to {}".format(
                location, default_location))
            myAlma.cache_location = default_location
    elif (location == default_location) and not os.path.isdir(location):  # create the 'data' subdirectory
        os.makedirs(default_location)
    if fitsonly:
        data_table = myAlma.get_data_info(uids_list, expand_tarfiles=True)
        # filter the data_table and keep only rows with "fits" in 'access_url' and the strings provided by user
        # in 'filename_must_include' parameter
        dl_table = data_table[[i for i, v in enumerate(data_table['access_url']) if v.endswith(".fits") and
                               all(i in v for i in filename_must_include)]]
    else:
        data_table = myAlma.get_data_info(uids_list, expand_tarfiles=False)
        # filter the data_table and keep only rows with "fits" in 'access_url' and the strings provided by user
        # in 'filename_must_include' parameter
        dl_table = data_table[[i for i, v in enumerate(data_table['access_url']) if
                               all(i in v for i in filename_must_include)]]
    dl_df = dl_table.to_pandas()
    # remove empty elements in the access_url column
    dl_df = dl_df.loc[dl_df.access_url != '']
    ####################### Below is to extract random fits files from the fits file list for all the matching observations #####################
    #######################                       Considerably slower but with greater variance                             #####################
    # pick out n random elements from dataframe to download
    dl_df_n_random = dl_df.sample(n=amount_of_random_files)
    dl_link_list = list(dl_df_n_random['access_url'].unique())
    # keep track of the download size and number of files to download
    dl_size = dl_df_n_random['content_length'].sum()
    dl_files = len(dl_df_n_random['access_url'].unique())
    dl_uid_list = list(dl_df_n_random['ID'].unique())

    if dryrun:
        print("This is a dryrun. To begin download, set dryrun=False.")
        print("================================")
    else:
        print("Starting download. Please wait...")
        print("================================")
        try:
            myAlma.download_files(dl_link_list, cache=True)
        except ValueError as e:
            print(e)
    if dl_files > 0:
        print("Download location = {}".format(myAlma.cache_location))
        print("Total number of Member OUSs to download = {}".format(len(dl_uid_list)))
        print("Selected Member OUSs: {}".format(dl_uid_list))
        print("Number of files to download = {}".format(dl_files))
        dl_size_fmt, dl_format = _format_bytes(dl_size)
        print("Needed disk space = {:.1f} {}".format(dl_size_fmt, dl_format))
        if print_urls:
            print("File URLs to download = {}".format("\n".join(dl_link_list)))
    else:
        print("Nothing to download.")
        print("Note: often only a subset of the observations (e.g. the representative window) is ingested into "
              "the archive. In such cases, you may need to download the raw dataset, reproduce the calibrated "
              "measurement set, and image the observations of interest. It is also possible to request calibrated "
              "measurement sets through a Helpdesk ticket to the European ARC "
              "(see https://almascience.eso.org/local-news/requesting-calibrated-measurement-sets-in-europe).")
    print("--------------------------------")

    # From here on there's download, we only want to download random amount of X declared when calling function
