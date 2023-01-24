from alminer.constants import VALID_KEYWORDS_STR
from alminer import filter_results
from pyvo.dal import tap
import re

def _set_service():
    """Set the url for the ALMA TAP service."""
    service = tap.TAPService("https://almascience.eso.org/tap")
    return service

def run_query_mod(query_str):
    """
    Run the TAP query through PyVO service.

    Parameters
    ----------
    query_str : str
         ADQL query to send to the PyVO TAP service

    Returns
    -------
    pandas.DataFrame containing the query results

    """
    service = _set_service()
    # Run query
    pyvo_TAP_results = service.search(query_str, maxrec=1000000)  # for large queries add maxrec=1000000
    # Transform output into astropy table first, then to a pandas DataFrame
    TAP_df = pyvo_TAP_results.to_table().to_pandas()
    # the column publication_year must be in 'object' type because it contains numbers and NaNs
    TAP_df['publication_year'] = TAP_df['publication_year'].astype('object')
    return TAP_df

def keysearch_mod(search_dict, public=True, published=None, print_query=False, print_targets=True):
    """
    Query the ALMA archive for any (string-type) keywords defined in ALMA TAP system.

    Parameters
    ----------
    search_dict : dict[str, list of str]
         Dictionary of keywords in the ALMA archive and their values. Values must be formatted as a list.
         A list of valid keywords are stored in VALID_KEYWORDS_STR variable.
    public : bool, optional
         (Default value = True)
         Search for public data (public=True), proprietary data (public=False),
         or both public and proprietary data (public=None).
    published : bool, optional
         (Default value = None)
         Search for published data only (published=True), unpublished data only (published=False),
         or both published and unpublished data (published=None).
    print_query : bool, optional
         (Default value = True)
         Print the ADQL TAP query to the terminal.
    print_targets : bool, optional
         (Default value = False)
         Print a list of targets with ALMA data (ALMA source names) to the terminal.

    Returns
    -------
    pandas.DataFrame containing the query results.

    Notes
    -----
    The power of this function is in combining keywords. When multiple keywords are provided, they are
    queried using 'AND' logic, but when multiple values are provided for a given keyword, they are queried using
    'OR' logic. If a given value contains spaces, its constituents are queried using 'AND' logic. The only exception
    to this rule is the 'target_name' keyword.

    Examples
    --------
    keysearch({"proposal_abstract": ["high-mass star formation outflow disk"]})
         will query the archive for projects with the words
         "high-mass" AND "star" AND "formation" AND "outflow" AND "disk" in their proposal abstracts.

    keysearch({"proposal_abstract": ["high-mass", "star", "formation", "outflow", "disk"]})
         will query the archive for projects with the words
         "high-mass" OR "star" OR "formation" OR "outflow" OR "disk" in their proposal abstracts.

    keysearch({"proposal_abstract": ["star formation"], "scientific_category":['Galaxies']})
         will query the archive for projects with the words
         "star" AND "formation" in their proposal abstracts AND
         projects that are within the scientific_category of 'Galaxies'.

    """
    print("================================")
    print("alminer.keysearch results ")
    print("================================")
    # Add keyword to the query dictionary for the data rights (Public, Proprietary, or both)
    if public:
        search_dict['data_rights'] = ['Public']
    elif not public and public is not None:
        search_dict['data_rights'] = ['Proprietary']
    # Add scan intent keyword to the query dictionary to be the science target by default
    search_dict['scan_intent'] = ['TARGET']

    # compile a list of queries based on all keywords provided
    full_query_list = []
    for keyword, values in search_dict.items():
        # Catch if a wrong keyword is used and give appropriate error
        assert keyword in VALID_KEYWORDS_STR, "Invalid keyword, must be one of: {}".format(VALID_KEYWORDS_STR)
        # Convert underscores and spaces in the target name to wildcard
        # target_name is always queried with OR logic
        if keyword == 'target_name':
            values = [v.replace('_', '%') for v in values]
            values = [v.replace(' ', '%') for v in values]
            # Create queries for a given keyword using 'OR' logic between different values and accounting for
            # the case-sensitivity
            current_query = ["LOWER({}) LIKE '%{}%'".format(keyword, v.lower()) for v in values]
            full_query_list.append("({})".format(" OR ".join(current_query)))
        # Account for AND/OR logic for keywords that are not target_name
        else:
            keyword_query_list = []
            for v in values:
                # If there are spaces in the values of a given keyword, split them out and query them with AND logic
                if re.search(r"\s", v):
                    split_values = v.split()
                    current_query = ["LOWER({}) LIKE '%{}%'".format(keyword, s.lower()) for s in split_values]
                    keyword_query_list.append("({})".format(" AND ".join(current_query)))
                # If separate words are provided as values, query them with OR logic
                else:
                    keyword_query_list.append("LOWER({}) LIKE '%{}%'".format(keyword, v.lower()))
            full_query_list.append("({})".format(" OR ".join(keyword_query_list)))
    # Put together the entire query with 'AND' logic between different keywords
    full_query = "SELECT * FROM ivoa.obscore WHERE {} ORDER BY proposal_id".format(" AND ".join(full_query_list))
    if print_query:
        print("Your query is: {}".format(full_query))
    TAP_df = run_query_mod(full_query)
    # Filter whether the user wants published data, unpublished data, or both (default)
    if published:  # case pf published = True
        TAP_df = TAP_df[TAP_df['publication_year'].notnull()]
    elif not published and published is not None:  # case pf published = False
        TAP_df = TAP_df[TAP_df['publication_year'].isnull()]
    return filter_results(TAP_df, print_targets=print_targets)