import alminer

# lowmass = alminer.keysearch({'science_keyword': ['"low-mass star formation"']})
# astrochemistry = alminer.keysearch({'science_keyword': ['"Astrochemistry"']})
intermass = alminer.keysearch(
    {'science_keyword': ['"intermediate-mass star formation"']})
outflows_feedback = alminer.keysearch(
    {'science_keyword': ['"outflows, jets, feedback"']})
outflows_winds = alminer.keysearch(
    {'science_keyword': ['"outflows, jets and ionized winds"']})
ism = alminer.keysearch(
    {'science_keyword': ['"inter-stellar medium (ism)/molecular clouds"']})

search_list = [intermass, outflows_feedback, outflows_winds, ism]

for i in search_list:
    alminer.download_data(i, fitsonly=True, dryrun=True, location='./data', filename_must_include=[
                          '.pbcor', 'cont', '_sci'], print_urls=True, archive_mirror='ESO')
