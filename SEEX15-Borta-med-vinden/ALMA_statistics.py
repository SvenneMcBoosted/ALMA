from datetime import date
from tabnanny import filename_only
import alminer
import alminer_extensions
import pandas
import alminer_extensions
import matplotlib
import numpy
import matplotlib.pyplot as plt
import os
import FittingData



# this changes the default date converter for better interactive plotting of dates:
plt.rcParams['date.converter'] = 'concise'


#query = alminer.keysearch({'science_keyword': ['Debri disks']})
query = alminer.keysearch({'science_keyword': ['Outflows, jets and ionized winds']})
#query = alminer.keysearch({'target_name': ['HATLAS_RED_2525']})
#query = alminer.keysearch({'target_name': ['UDst3_pKIRAC_0009']})
#lägg till en query som tas från main så att man kan få statistik från de sökningar man gör i huvudprogrammet.
#query = (initquery[initquery.filename_only])



def observations_by_year(query):

    #the years that are being checked
    categories = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
    data = []

    #going through the query and counting amount of observations each year
    for i in categories:
        count = 0
        queryData = (query[query.obs_release_date > i])
        queryData = (queryData[queryData.obs_release_date < str(int(i)+1)])

        for i in range(len(queryData)):
            count += 1
        data.append(count)

    #prints figure showing the data
    fig1, ax = plt.subplots(figsize=(8, 4), layout='constrained') 
    fig1.canvas.manager.set_window_title('Publikationsår för observationerna i sökningen')
    ax.bar(categories, data)



#statistics about what electon transitions are covered by observations 
def electron_transitions(query, frequencies, z=0., only_relevant=True):
    line_names = (frequencies['Species'] + ' ' + frequencies['Resolved QNs']).tolist()
    line_freqs = frequencies['Ordered Freq (GHz)'].tolist()
    minfreq = min(query['min_freq_GHz'].tolist())
    maxfreq = max(query['max_freq_GHz'].tolist())
    categories = []
    data = []
    for t, line in enumerate(line_names):
        if not (minfreq <= line_freqs[t] <= maxfreq) and only_relevant:
            continue
        line_df = alminer.line_coverage(query, line_freq=line_freqs[t], z=z, line_name=line_names[t], print_summary=False, print_targets=False)
        if not line_df.empty:
            categories.append(line_names[t])
            data.append(len(line_df))

    fig2, bx = plt.subplots(figsize=(12, 8)) 
    fig2.canvas.manager.set_window_title('Molekyler och elektronövergångar i sökningen')
    y_pos = range(len(categories))
    plt.xticks(y_pos, shorten_line_names(categories), rotation=90)
    bx.bar(shorten_line_names(categories), data)

# skymap för vart alla observationer från arkivet är
def fig_skymap(query):
    alminer.plot_sky(query)


def shorten_line_names(names):
    newNames = []
    for name in names:
        newNames.append(name.replace("J=","").replace("v=0","").split(",F=")[0])
    return newNames


def missing_cont_file(query):
    total_obs = 0
    has_cont = 0
    for i in range(len(query)):
        line = query.take([i])
        total_obs += 1
        #data = 0
        data = alminer.download_data(line, fitsonly=False, dryrun=True, print_urls=False, filename_must_include=".cont")
        print(data)
        if data != None:
            has_cont += 1
        percentage = has_cont/total_obs * 100
        print(str(has_cont) + " observation(er) inehåller en .cont fil utav totalt: " + str(total_obs) + " observation(er)." + " (" + str(percentage) + "%)")

## fix this function, use functions in fitting data that is actually used to analyse data
def analasys_success(total_files_analysed, total_files_analysed_2, analasys_failed, analasys_failed_2):
    analasys_fig, dx = plt.subplots(figsize=(9, 5), layout='constrained') 
    analasys_fig.canvas.manager.set_window_title('Mängden framgångsrika analyser i sökningen')
    #y_pos = range(len(categories))
    #plt.xticks(y_pos, , rotation=90)
    categories_analasys_success = ["Analyser", "Misslyckade analyser","Analyser 2", "Misslyckade analyser 2"]
    data_analasys_success = [total_files_analysed, analasys_failed, total_files_analysed_2, analasys_failed_2]
    dx.bar(categories_analasys_success, data_analasys_success)


#amount of observations by year #DONE#
#observations_by_year(query)

# stats of electron transitions #DONE#
#electron_transitions(query, alminer_extensions.get_frequencies("molecules.csv"))

#Skymap of all observations in a query #DONE#
fig_skymap(query)

#Amount of failed analasys
#analasys_success(FittingData.total_files_analysed, FittingData.total_files_analysed_2, FittingData.analasys_failed, FittingData.analasys_failed_2)
#analasys_success(20, 16, 7, 2)
#Number of observations that are missing cont files #DONE# has to use dryruns but think that might be the only way to do it.
#missing_cont_file(query)

plt.show()

