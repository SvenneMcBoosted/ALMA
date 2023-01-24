from os import system
import alminer
import pandas as pd
from astroquery.alma import Alma
from astropy.io import fits
import numpy as np
import os


def get_freq(freqs):
    res = freqs.split(',')
    return float(res[0])


# Takes in frequencies from a splatalogue CSV file and returns a formated pandas data frame
def get_frequencies(dir):
    csv_file = pd.read_csv(dir, sep=':')
    csv_file = csv_file.drop(['Chemical Name', 'CDMS/JPL Intensity', 'Lovas/AST Intensity',
                              'E_L (cm^-1)', 'E_L (K)', 'Linelist'], axis=1)
    csv_file = csv_file.rename(columns={'Ordered Freq (GHz) (rest frame, redshifted)': 'Ordered Freq (GHz)'})
    csv_file['Ordered Freq (GHz)'] = csv_file['Ordered Freq (GHz)'].apply(get_freq)
    return csv_file[csv_file['Resolved QNs'].str.contains('F') == False]

# Takes dataframe returned by get_frequencies and a given frequency range and matches to a 
# molecule, if no direct match gives closest match
def get_molecule(ref_freqs, min_freq, max_freq):
    smallest_diff = 1000000000000 # 1000 GHz
    closest_match = None
    mid_freq = (max_freq + min_freq) / 2

    for index, row in ref_freqs.iterrows():
        tmp = min(abs(row['Ordered Freq (GHz)']*1000000000 - min_freq), abs(row['Ordered Freq (GHz)']*1000000000 - max_freq))
        if tmp < smallest_diff:
            smallest_diff = tmp
            closest_match = row['Species'] + ' ' + row['Resolved QNs']
        # Below should probably be tmp < 0.1 or something
        if min_freq <= row['Ordered Freq (GHz)']*1000000000 <= max_freq:
            return row['Species'] + ' ' + row['Resolved QNs']
    if closest_match is None:
        closest_match = 'No match found'
    return 'Closest match: ' + closest_match


# Generalized lines function. frequencies takes the outputted dataframe from get_frequencies
def get_lines(observations, frequencies, z=0., only_relevant=True, print_summary=True, print_targets=True):
    df_list = []
    line_names = (frequencies['Species'] + ' ' + frequencies['Resolved QNs']).tolist()
    line_freqs = frequencies['Ordered Freq (GHz)'].tolist()
    minfreq = min(observations['min_freq_GHz'].tolist())
    maxfreq = max(observations['max_freq_GHz'].tolist())

    for t, line in enumerate(line_names):
        if not (minfreq <= line_freqs[t] <= maxfreq) and only_relevant:
            continue
        line_df = alminer.line_coverage(observations, line_freq=line_freqs[t], z=z,
                                        line_name=line_names[t], print_summary=print_summary,
                                        print_targets=print_targets)
        if not line_df.empty:
            df_list.append(line_df)
    if df_list:
        df = pd.concat(df_list)
        # need to reset the index of DataFrame so the indices in the final DataFrame are consecutive
        df = df.drop_duplicates().reset_index(drop=True)
        return df
    else:
        print("Found no ALMA observations covering transitions of given molecules.")
        print("--------------------------------")


SiO_line_names = ["SiO (1-0)", "SiO (2-1)", "SiO (3-2)", "SiO (4-3)", "SiO (5-4)", "SiO (6-5)",
                  "SiO (7-6)", "SiO (8-7)", "SiO (9-8)", "SiO (10-9)", "SiO (11-10)", "SiO (12-11)"]
SiO_line_freq = {"SiO (1-0)": 43.42376000, "SiO (2-1)": 86.84696000, "SiO (3-2)": 130.26861000,
                 "SiO (4-3)": 173.68831000, "SiO (5-4)": 217.10498000, "SiO (6-5)": 260.51802000,
                 "SiO (7-6)": 303.92696000, "SiO (8-7)": 347.33063100, "SiO (9-8)": 390.72844830,
                 "SiO (10-9)": 434.11955210, "SiO (11-10)": 477.50309650, "SiO (12-11)": 520.87820390}


def SiO_lines(observations, z=0., print_summary=True, print_targets=True):
    SiO_df_list = []
    for t, line in enumerate(SiO_line_names):
        line_df = alminer.line_coverage(observations, line_freq=SiO_line_freq[line], z=z,
                                        line_name=SiO_line_names[t], print_summary=print_summary,
                                        print_targets=print_targets)
        if not line_df.empty:
            SiO_df_list.append(line_df)
    if SiO_df_list:
        SiO_df = pd.concat(SiO_df_list)
        # need to reset the index of DataFrame so the indices in the final DataFrame are consecutive
        SiO_df = SiO_df.drop_duplicates().reset_index(drop=True)
        return SiO_df
    else:
        print("Found no ALMA observations covering transitions of SiO.")
        print("--------------------------------")


CS_line_names = ["C34S (1-0)", "C34S (2-1)", "C34S (3-2)", "C34S (4-3)", "C34S (5-4)", "C34S (6-5)",
                 "C34S (7-6)", "C34S (8-7)", "C34S (9-8)", "C34S (10-9)", "C34S (11-10)", "C34S (12-11)"]
CS_line_freq = {"C34S (1-0)": 48.20694110, "C34S (2-1)": 96.41294950, "C34S (3-2)": 144.61710070,
                "C34S (4-3)": 192.81845660, "C34S (5-4)": 241.01608920, "C34S (6-5)": 289.20906840,
                "C34S (7-6)": 337.39645900, "C34S (8-7)": 385.57733600, "C34S (9-8)": 433.75076300,
                "C34S (10-9)": 481.91581050, "C34S (11-10)": 530.07155370, "C34S (12-11)": 578.21706900}


def CS_lines(observations, z=0., print_summary=True, print_targets=True):
    CS_df_list = []
    for t, line in enumerate(CS_line_names):
        line_df = alminer.line_coverage(observations, line_freq=CS_line_freq[line], z=z,
                                        line_name=CS_line_names[t], print_summary=print_summary,
                                        print_targets=print_targets)
        if not line_df.empty:
            CS_df_list.append(line_df)
    if CS_df_list:
        CS_df = pd.concat(CS_df_list)
        # need to reset the index of DataFrame so the indices in the final DataFrame are consecutive
        CS_df = CS_df.drop_duplicates().reset_index(drop=True)
        return CS_df
    else:
        print("Found no ALMA observations covering transitions of C34S.")
        print("--------------------------------")

# removes all projects which do not include any of the rotational transitions we want to study.
def removeAllProjectsWithoutMolecules(dataframe, frequencies):
    trueFalse = [moleculesInRange(minFreq, maxFreq, frequencies) for minFreq, maxFreq in
                 zip(dataframe['min_freq_GHz'], dataframe['max_freq_GHz'])]
    dataframe = dataframe.reset_index()
    dataframe = dataframe.drop(trueFalseToIndex(trueFalse)) # Think we can just do something like dataframe = dataframe[trueFalse]
    return dataframe

# checks if there are any transitions of interest in the range.
def moleculesInRange(min, max, frequencies):
    for freq in frequencies["Ordered Freq (GHz)"].tolist():
        if min < freq < max:
            return True
    return False

# turns an array of booleans to an array of indices to remove (False = Remove)
def trueFalseToIndex(TrueFalse): # this is kinda whack
    indicies = []
    for i in range(len(TrueFalse)):
        if not TrueFalse[i]:
            indicies.append(i)
    return indicies


# Same as alminer.download_data except it ignores individual files larger than {maxSize} GB
# TODO: testa molekylfiltret mer
def download_data2(observations, fitsonly=False, dryrun=False, print_urls=False, filename_must_include='',
                   location='./data', frequencies=[], maxSize=20):
    """
    Download ALMA data from the archive to a location on the local machine.
    Parameters
    ----------
    observations : pandas.DataFrame
         This is likely the output of e.g. 'conesearch', 'target', 'catalog', & 'keysearch' functions.
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
    frequencies: dataframe
         Dataframe of frequencies from splatalogue as given by alminer_extentions.get_frequencies()
    maxSize : float
         (Default value = 20)
         The maximum file size of a single file in GB.

    """
    print("================================")
    # we use astroquery to download data
    myAlma = Alma()
    default_location = './data'
    myAlma.cache_location = default_location
    # catch the case where the DataFrame is empty.
    try:
        if any(observations['data_rights'] == 'Proprietary'):
            print("Warning: some of the data you are trying to download are still in the proprietary period and are "
                  "not publicly available yet.")
            observations = observations[observations['data_rights'] == 'Public']
        uids_list = observations['member_ous_uid'].unique()
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
    # change download location if specified by user, else the location will be the astrquery cache location
    if location != default_location:
        if os.path.isdir(location):
            myAlma.cache_location = location
        else:
            print("{} is not a directory. The download location will be set to {}".format(location, default_location))
            myAlma.cache_location = default_location
    if fitsonly:
        data_table = Alma.get_data_info(uids_list, expand_tarfiles=True)
        # filter the data_table and keep only rows with "fits" in 'access_url' and the strings provided by user
        # in 'filename_must_include' parameter
        dl_table = data_table[[i for i, v in enumerate(data_table['access_url']) if v.endswith(".fits") and
                               all(i in v for i in filename_must_include)]]
        #dl_table.pprint_all()

        # General idea is to check the mfs file to be able to map spw to frequencies so that we can filter out and only
        # download cube files of the transitions we are interested in.
        oldLength = len(dl_table)
        if ".cube" in filename_must_include and len(frequencies) > 0:
            UIDquery = alminer.keysearch({'member_ous_uid': [uids_list[0]]})
            alminer.download_data(UIDquery, fitsonly=True, filename_must_include=["_sci", ".pbcor", ".mfs"],
                                  location=location)
            i=0
            spwToRestFreq = []
            for filename in os.listdir(location):
                if filename.__contains__(".mfs"):
                    with fits.open(location + "/" + filename) as hdul:
                        h = hdul[0].header
                        freq = h.get("RESTFRQ")
                        spw = h.get("SPW")
                        spwToRestFreq.append([spw, freq,i])
                        i = i+1
                        del hdul[0].data

            spwToRestFreq = np.array(sorted(spwToRestFreq, key = lambda x: x[1])) # sort according to frequencies

            #print(spwToRestFreq)

            newIndex = spwToRestFreq[:,2] # spw order
            #print(newIndex)

            UIDquery = UIDquery.sort_values(by=['min_freq_GHz'])
            UIDquery = UIDquery.reset_index()

            UIDquery["newIndex"] = newIndex
            UIDquery = UIDquery.set_index("newIndex")
            UIDquery = UIDquery.sort_values(by=["newIndex"]) # sort the query in "spw order".

            #UIDquery.to_csv("query2.csv")
            trueFalse = [moleculesInRange(minFreq, maxFreq, frequencies) for minFreq, maxFreq in
                         zip(UIDquery['min_freq_GHz'], UIDquery['max_freq_GHz'])]
            dl_table = dl_table[trueFalse]
            #dl_table.pprint_all()

        # dl_table.pprint_all()
        dl_table = dl_table[dl_table['content_length'] < maxSize * 1e9] # filter on file size (hard to handle too large files)
        dl_link_list = dl_table['access_url'].tolist()
        # keep track of the download size and number of files to download
        dl_size = dl_table['content_length'].sum() / 1E9
        dl_files = len(dl_table)
        print("Original number of files:" +oldLength + ". Reduced number of files: " + dl_files+".")
        if dryrun:
            print("This is a dryrun. To begin download, set dryrun=False.")
            print("================================")
        else:
            print("Starting download. Please wait...")
            print("================================")
            myAlma.download_files(dl_link_list, cache=True)
    else:
        data_table = Alma.get_data_info(uids_list, expand_tarfiles=False)
        dl_link_list = data_table['access_url'].tolist()
        # keep track of the download size and number of files to download
        dl_size = data_table['content_length'].sum() / 1E9
        dl_files = len(data_table)
        if dryrun:
            print("This is a dryrun. To begin download, set dryrun=False.")
            print("================================")
        else:
            print("Starting download. Please wait...")
            print("================================")
            myAlma.retrieve_data_from_uid(uids_list, cache=True)
    print("Download location = {}".format(myAlma.cache_location))
    print("Total number of Member OUSs to download = {}".format(len(uids_list)))
    print("Selected Member OUSs: {}".format(uids_list.tolist()))
    print("Number of files to download = {}".format(dl_files))
    if dl_size > 1000.:
        print("Needed disk space = {:.1f} TB".format(dl_size / 1000.))
    elif dl_size < 1.:
        print("Needed disk space = {:.1f} MB".format(dl_size * 1000.))
    else:
        print("Needed disk space = {:.1f} GB".format(dl_size))
    if print_urls:
        print("File URLs to download = {}".format("\n".join(dl_link_list)))
    print("--------------------------------")
