import string
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import upsetplot
from datetime import datetime


def set_seed(seed):
    """
    Set random seeds to generate values for dataset.
    """
    np.random.seed(seed)
    random.seed(seed)


def generate_pattern(
    pattern, num_rows, num_cols, num_int, type="str", seed=42
):
    set_seed(seed)
    if pattern == "monotone":
        df = generate_monotone(num_rows, num_cols, num_int, type)
    elif pattern == "general":
        df = generate_general(num_rows, num_cols, num_int, type)
    else:
        df = pd.DataFrame()
    return df


def generate_monotone(num_rows, num_cols, num_int, data_type):
    """Generate a data frame that has columns of type 'object',
    and a monotone pattern of missing data


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
            non_missing_values = [get_value(data_type)] * (
                num_cols - num_missing
            )
            rows.append(non_missing_values + [np.nan] * num_missing)

    cols = ["v" + str(i) for i in range(num_cols)]
    df = pd.DataFrame(rows, columns=cols, dtype=object)

    return df


def generate_general(num_rows, num_cols, num_int, data_type):
    """Generate a data frame that has columns of type 'object',
    and a general pattern of missing data.


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
    for _ in range(num_rows):
        if count == num_int:  # reset counter to repeat patterns
            count = 0
        intersection = intersections_template[count, :]
        values = np.where(intersection, get_value(type), np.nan)
        rows.append(values)
        count += 1
    cols = ["v" + str(i) for i in range(num_cols)]
    return pd.DataFrame(rows, columns=cols, dtype=object)


def get_value(data_type="str"):
    if data_type == "str":
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=8)
        )
    elif data_type == "float":
        return np.random.random()
    else:
        return -999


def compute_missingness(df):
    """Compute missingness from data based on package provided."""
    return upsetplot.from_indicators(indicators=pd.isna, data=df)


def eval_data(df):
    """
    Evaluates the performance of UpSet by timing the
    missingness computation and the visualisation of the provided data.

    Parameters
    ----------
    df : pd.DataFrame
        data frame

    Returns
    -------
    results : float
        time [ms] how long it takes to compute missingness and visualise it

    """
    try:
        time1 = datetime.now()
        data_missing = compute_missingness(df)

        # TODO: add that we write plot to png or some other file for timing
        upsetplot.plot(data_missing, show_counts=True)
        # plt.show(block=False)
        plt.savefig("tmp.pdf")
        time2 = datetime.now()

        return str(time2 - time1)

    except:
        raise
