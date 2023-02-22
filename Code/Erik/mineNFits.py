"""Download specified amount of fits files. Change amount_of_random_files to choose how many.
   Currently, the download searches through all of the matching observations and randomly
   pick 20 fits files from these observations. It takes a while, ish 45 minutes on my computer."""

import alminer
import downloadData

# Astrochemistry doesn't seem to work either with proposal_abstract or science_keyword
# astrochemistry = alminer.keysearch({'proposal_abstract': ['"Astrochemistry"']})
# astrochemistry = alminer.keysearch({'science_keyword': ['"Astrochemistry"']})

# Science keyword search for the rest of the interesting keywords
lowmass = alminer.keysearch(
    {'science_keyword': ['"disks around low-mass stars"']})
intermass = alminer.keysearch(
    {'science_keyword': ['"intermediate-mass star formation"']})
outflows_feedback = alminer.keysearch(
    {'science_keyword': ['"outflows, jets, feedback"']})
outflows_winds = alminer.keysearch(
    {'science_keyword': ['"outflows, jets and ionized winds"']})
ism = alminer.keysearch(
    {'science_keyword': ['"inter-stellar medium (ism)/molecular clouds"']})

search_dict = {'lowmass': lowmass, 'intermass': intermass,
               'outflows_feedback': outflows_feedback, 'outflows_winds': outflows_winds, 'ism': ism}

# downloadData.download_n_data(lowmass, amount_of_random_files=15, fitsonly=True, dryrun=False, location='./data',
#                             filename_must_include=['.pbcor', 'cont', '_sci'], print_urls=True, archive_mirror='ESO')

# Downloading 15 fits files from each category (dryrun=True to not download)
for key, value in search_dict.items():
    downloadData.download_n_data(value, amount_of_random_files=20, fitsonly=True, dryrun=False, location='./data/{}'.format(key), filename_must_include=[
        '.pbcor', 'cont', '_sci'], print_urls=True, archive_mirror='ESO')
