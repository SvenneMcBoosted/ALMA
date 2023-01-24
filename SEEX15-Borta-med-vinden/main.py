from gc import get_freeze_count
import alminer
import alminer_extensions as almext
from keysearchmod import keysearch_mod
import pandas
import sys
import os
from astropy.io import ascii
from soupsieve import select
from FittingData import analyseDir, analyseDir2
import time

frequencies = almext.get_frequencies('./molecules.csv')

def download_routine(datadir, dryrun, keywords):
    for i in range(len(keywords)):
            # Query and filtering
            print("Querying with keyword: " + keywords[i])
            my_query = alminer.keysearch({'science_keyword':[keywords[i]]}, print_targets=False)
            selected = my_query[my_query.obs_release_date > '2016']
            selected = selected[selected.ang_res_arcsec < 0.4] # Based on what Per said
            selected = almext.removeAllProjectsWithoutMolecules(selected,frequencies)
            selected = selected.drop_duplicates(subset='obs_id').reset_index(drop=True)
            selected = selected.sort_values(by=['obs_release_date'], ascending=False)
            print(len(selected))

            # Iterates over the rows
            for i in range(len(selected)):
                tmp = selected.take([i])

                obsdir = datadir + "/" + selected.iloc[i].at['obs_id'].replace('uid://','').replace('/','-')

                if os.path.isdir(obsdir):
                    print("Already analysed, skipping")
                    continue

                if dryrun == 'True':
                    almext.download_data2(tmp, fitsonly=True, dryrun=True, location=datadir, filename_must_include=[".pbcor", "_sci"], maxSize=30)
                while(True):
                    inp = input("Would you like to proceed with the download? [y/n]: ")
                    #inp = "y"
                    if inp.lower() == 'y':
                        os.mkdir(obsdir)
                        almext.download_data2(tmp, fitsonly=True, dryrun=False, location=obsdir, filename_must_include=[".pbcor", "_sci", ".cont"], maxSize=30)
                        hasCont = False
                        for filename in os.listdir(obsdir):
                            if filename.__contains__(".cont"):
                                hasCont = True

                        if hasCont:
                            almext.download_data2(tmp, fitsonly=True, dryrun=False, location=obsdir, filename_must_include=[".pbcor", "_sci", ".cube"], maxSize=30,frequencies=frequencies)
                            analyseDir2(obsdir)
                            time.sleep(20) #idk
                            deleteAllFits(obsdir)
                        break
                    elif inp.lower() == 'n':
                        print("Ok, skipping.")
                        break
                    else:
                        print('Incorrect input, try again.')

def deleteAllFits(dir):
    for filename in os.listdir(dir):
        if filename.__contains__(".fits"):
            os.remove(dir + "/" + filename)


# main program. 
# sys.argv[1] == 'all' for all keywords, or a single keyword
# sys.argv[2] == True for dryrun before download, otherwise it can be anything
def main():
    # Directory for observation data
    datadir = './data'
    # Terminal inputs, given default values in case none are given
    arg1 = 'all'
    arg2 = 'True'

    # All our chosen keywords
    keywords = ['Disks around low-mass stars', 'Debris disks', 'Disks around high-mass stars', 'Astrochemistry',
                'Giant Molecular Clouds (GMC) properties', 'High-mass star formation', 'Exo-planets', 
                'Inter-Stellar Medium (ISM)/Molecular clouds', 'HII regions', 'Intermediate-mass star formation', 
                'Low-mass star formation', 'Outflows, jets and ionized winds', 
                'Photon-Dominated Regions (PDR)/X-Ray Dominated Regions (XDR)', 'Pre-stellar cores']
    #categories = ['Disks and planet formation', 'ISM and star formation', 'Stars and stellar evolution']
    
    #Checks amount of terminal arguments
    if len(sys.argv) >= 2:
        arg1 = sys.argv[1]
    if len(sys.argv) >= 3:
        arg2 = sys.argv[2]
    # Checks that the terminal input is correct
    if arg1 not in keywords and arg1.lower() != 'all':
        print("Incorrect input, shutting down.")
        quit()

    # If sys.argv[1] == 'all' then keywords consists of all keywords, otherwise takes the terminal input (a single keyword)
    keywords = keywords if arg1.lower() == 'all' else [arg1]
    # Makes a folder for data downloads if there is none
    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    download_routine(datadir, arg2, keywords)

    print("Program finished, shutting down.")

if __name__ == "__main__":
    main()
