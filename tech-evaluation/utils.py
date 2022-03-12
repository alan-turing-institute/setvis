import string
import random
import pandas as pd
import numpy as np
import psutil
from datetime import datetime
import tempfile
from matplotlib import pyplot as plt
from bokeh.io import export_svg

import upsetplot
from pace.membership import Membership
from pace.plots import PlotSession


def set_seed(seed=42):
    """
    Set random seeds to generate values for dataset.
    """
    np.random.seed(seed)
    random.seed(seed)


def generate_pattern(pattern, num_rows, num_cols, num_int, type="str"):

    if pattern == "monotone":
        df = generate_monotone(num_rows, num_cols, num_int, type)
    elif pattern == "general":
        df = generate_general(num_rows, num_cols, num_int, type)
    else:
        df = pd.DataFrame()
    return df


def generate_monotone(num_rows, num_cols, num_int, type):
    """
    Generate a data frame that has columns of type 'object', and a monotone pattern of missing data
    

    Parameters
    ----------
    num_cols : int
        The number of columns in the data frame
    num_rows : int
        The number of rows in the data frame
    num_int : int
        The number of unique intersections
    type : str
        The data type of the records

    Returns
    -------
    df : pd.DataFrame
        Data frame containing the dataset.

    """
    rows = []
    if num_int > num_cols:
        num_int = num_cols
    for i in range(num_rows):
        num_missing = num_cols - (i % num_cols)

        if num_missing == num_cols:
            rows.append([np.nan] * num_missing)
        else:
            # non_missing_values = [
            #     "".join(
            #         random.choices(string.ascii_letters + string.digits, k=8)
            #     )
            # ] * (num_cols - num_missing)
            non_missing_values = [get_value(type)] * (num_cols - num_missing)
            rows.append(non_missing_values + [np.nan] * num_missing)

    cols = ["v" + str(i) for i in range(num_cols)]
    df = pd.DataFrame(rows, columns=cols, dtype=object)

    return df


def generate_general(num_rows, num_cols, num_int, type):
    """
    Generate a data frame that has columns of type 'object', and a general pattern of missing data.
    

    Parameters
    ----------
    num_cols : int
        The number of columns in the data frame
    num_rows : int
        The number of rows in the data frame
    num_int : int
        The number of unique intersections

    Returns
    -------
    pd.DataFrame
        Data frame containing the dataset.

    """
    rows = []
    intersections_template = np.random.randint(
        2, size=(num_int, num_cols)
    )  # matrix of 0,1 of size nr_int x nr_col

    count = 0
    for i in range(num_rows):
        if count == num_int:  # reset counter to repeat patterns
            count = 0
        intersection = intersections_template[count, :]
        values = np.where(intersection, get_value(type), np.nan)
        rows.append(values)
        count += 1
    cols = ["v" + str(i) for i in range(num_cols)]
    return pd.DataFrame(rows, columns=cols, dtype=object)


def get_value(type="str"):
    if type == "str":
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=8)
        )
    elif type == "float":
        return np.random.random()
    else:
        return -999


def eval_data(df, package, pattern, num_rows, num_cols, num_int, type):
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
    type : str
        The data type of the records

    Returns 
    -------
    results : list
        ... with the results of the performance evaluation 
    
    """
    try:
        results = [
            [
                package,
                pattern,
                num_rows,
                num_cols,
                num_int,
                type,
                "START",
                None,
                psutil.virtual_memory(),
            ]
        ]
        # compute missingness
        start_time = datetime.now()
        if package == "upset":
            data_missing = upsetplot.from_indicators(
                indicators=pd.isna, data=df
            )
        elif package == "pace":
            data_missing = Membership.from_data_frame(df)
        time2 = datetime.now()
        td = time2 - start_time
        results.append(
            [
                package,
                pattern,
                num_rows,
                num_cols,
                num_int,
                type,
                "COMPUTE",
                td.seconds + td.microseconds / 1e6,
                psutil.virtual_memory(),
            ]
        )
        # visualisations
        time3 = datetime.now()
        # TODO: add that we write plot to png or some other file for timing
        if package == "upset":
            upsetplot.plot(data_missing, show_counts=True)
            # plt.show(block=False)
            time4 = datetime.now()
            plt.savefig("tmp.png")
        elif package == "pace":
            session = PlotSession(df)
            session.add_plot("a")
            # TODO: how can we access plot?

        td = time4 - time3
        results.append(
            [
                package,
                pattern,
                num_rows,
                num_cols,
                num_int,
                type,
                "VISUALIZE",
                td.seconds + td.microseconds / 1e6,
                psutil.virtual_memory(),
            ]
        )
        return results
    except:
        raise
