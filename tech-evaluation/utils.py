import string
import random
import pandas as pd
import numpy as np
import psutil
from datetime import datetime
import upsetplot
from matplotlib import pyplot as plt


def generate_pattern(pattern, num_rows, num_cols):
    if pattern == "monotone":
        df = generate_monotone(num_rows, num_cols)
    else:
        df = pd.DataFrame()
    return df


def generate_monotone(num_rows, num_cols):
    """
    Generate a data frame that has columns of type 'object', and a monotone pattern of missing data
    

    Parameters
    ----------
    num_cols : int
        The number of columns in the data frame
    num_rows : int
        The number of rows in the data frame

    Returns
    -------
    Data frame containing the dataset.

    """
    rows = []

    for i in range(num_rows):
        num_missing = num_cols - (i % num_cols)

        if num_missing == num_cols:
            rows.append([np.nan] * num_missing)
        else:
            non_missing_values = [
                "".join(
                    random.choices(string.ascii_letters + string.digits, k=6)
                )
            ] * (num_cols - num_missing)
            rows.append(non_missing_values + [np.nan] * num_missing)

    cols = ["v" + str(i) for i in range(num_cols)]
    df = pd.DataFrame(rows, columns=cols, dtype=object)

    return df


def eval_data(df, package, pattern, num_rows, num_cols):
    """
    Evaluates the performance of UpSet by timing the 
    missingness computation and the visualisation of the provided data.

    Parameters
    ----------
    df : pd.DataFrame
        data frame
    package : str
        name of the evaluated visualisation package 
    pattern : str
        name of the pattern used to generate data
    num_rows : int
        number of rows in the dataset (records)
    num_cols : int
        number of columns in the dataset
    Returns 
    -------
    results : 
        ... with the results of the performance evaluation 
    
    """
    try:
        results = [
            [
                package,
                pattern,
                num_rows,
                num_cols,
                "START",
                None,
                psutil.virtual_memory(),
            ]
        ]
        # compute missingness
        start_time = datetime.now()
        data_missing = upsetplot.from_indicators(indicators=pd.isna, data=df)
        time2 = datetime.now()
        td = time2 - start_time
        results.append(
            [
                package,
                pattern,
                num_rows,
                num_cols,
                "COMPUTE",
                td.seconds + td.microseconds / 1e6,
                psutil.virtual_memory(),
            ]
        )
        # visualisations
        time3 = datetime.now()
        upsetplot.plot(data_missing, show_counts=True)
        plt.show(block=False)
        time4 = datetime.now()
        td = time4 - time3
        results.append(
            [
                package,
                pattern,
                num_rows,
                num_cols,
                "VISUALIZE",
                td.seconds + td.microseconds / 1e6,
                psutil.virtual_memory(),
            ]
        )
        return results
    except:
        raise
