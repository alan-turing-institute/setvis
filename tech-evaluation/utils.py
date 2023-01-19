import string
import math
import random
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import upsetplot
from pace.membership import Membership


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


def generate_general_missing(num_rows, num_cols, num_combinations):
    """Generate a data frame that has columns of type 'object', and a general pattern of missing data.
    Each row will be missing at least one value.
    Each column to be missing by itself at least once.
    There to be at least one intersection of each possible length (2 - num_cols).
    Each combination will occur an equal number of times, to the extent that is possible
    The rows are shuffled so they are in a 'random' order.


    Parameters
    ----------
    num_cols : int
        The number of columns in the data frame
    num_rows : int
        The number of rows in the data frame
    num_combinations : int
        The number of combinations of missing values (num_rows >= num_combinations; num_combinations >= 2 * num_cols - 1)

    Returns
    -------
    pd.DataFrame
        Data frame containing the dataset.

    """
    # Calculate the maximum number of combinations of each length (1 - num_cols)
    max_comb = []
    # Create a list of lists that store [combination length, max possible combinations]
    length_ncomb = []
    for l1 in range(num_cols):
        max_comb.append(math.comb(num_cols, l1+1))
        
        if l1 > 0 and l1 < num_cols-1:
            # Don't calculate this for combination length = 1 or num_cols
            length_ncomb.append([l1+1, max_comb[-1]])
    
    # Sort the list of lists
    length_ncomb.sort(key=lambda x:x[1])
    
    if num_combinations <= sum(max_comb) and num_rows >= num_combinations and num_combinations >= num_cols + (num_cols - 1):
        # Calculate the number of combinations of each length (1 column, 2 columns, etc.) so the distribution is as even as possible
        # If you're interested 'esi' is an acronym for exclusive set intersections
        esi_params = {1: num_cols, num_cols: 1}
        ncombs = esi_params[1] + esi_params[num_cols]
        # Loop over the other combination lengths, leaving the one that can have the most combinations until last
        cnt = 0
        for item in length_ncomb:
            clen = item[0]
            esi_params[clen] = min(item[1], int((num_combinations - ncombs + len(length_ncomb) - cnt - 1) / (len(length_ncomb) - cnt)))
            ncombs = ncombs + esi_params[clen]
            cnt = cnt + 1

        intersections_template = np.ones(shape=(num_combinations, num_cols)) # matrix of ones
        
        # Loop over the parameter dictionary to create a list of lists. Each
        # item is a combination of missing values
        ii = 0
        for key, value in esi_params.items():
            # Randomise the order of the sets for ESIs with this cardinality
            columns = np.arange(1, num_cols+1)
            # The number of ESIs and transactions with this cardinality
            esi_count = 0
            # Loop until the specified number of ESIs has been generated
            for i in itertools.combinations(columns, key):
                for n in i:
                    intersections_template[ii][n-1] = np.nan
                    
                #print(intersections_template[ii, :])
                ii = ii + 1
                esi_count = esi_count + 1
                if esi_count == value:
                    break
                
        # Create a data frame that contains all the intersections, repeating them until there are the specified number of rows
        rows = []
    
        count = 0
        for _ in range(num_rows):
            if count == num_combinations:  # reset counter to repeat patterns
                count = 0
            rows.append(intersections_template[count, :])
            count += 1
        cols = ["v" + str(i) for i in range(num_cols)]
        df = pd.DataFrame(rows, columns=cols, dtype=object).sample(frac=1).reset_index(drop=True)
    else:
        # num_rows or num_combinations is too small to generate a general missing data pattern for num_cols columns
        print('ERROR', 'num_rows =', num_rows, 'num_cols =', num_cols, 'num_combinations =', num_combinations, 'min allowed =', num_cols*2-1, 'max_possible =', sum(max_comb))
        df = None
    
    return df


def generate_planned_missing(num_rows, num_cols):
    """Generate a data frame that has columns of type 'object',
    and a planned pattern of missing data (each row is missing a value in one
    column). The rows are shuffled so they are in a 'random' order.


    Parameters
    ----------
    num_cols : int
        The number of columns in the data frame
    num_rows : int
        The number of rows in the data frame

    Returns
    -------
    pd.DataFrame
        Data frame containing the dataset.

    """
    rows = []
    intersections_template = np.ones(shape=(num_cols, num_cols)) # matrix of 0,1 of size nr_int x nr_col
    # Flag the diagonal elements as missing
    for l1 in range(num_cols):
        intersections_template[l1][l1] = np.nan

    count = 0
    for _ in range(num_rows):
        if count == num_cols:  # reset counter to repeat patterns
            count = 0
        rows.append(intersections_template[count, :])
        count += 1
    cols = ["v" + str(i) for i in range(num_cols)]
    return pd.DataFrame(rows, columns=cols, dtype=object).sample(frac=1).reset_index(drop=True)


def get_value(data_type="str"):
    if data_type == "str":
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=8)
        )
    elif data_type == "float":
        return np.random.random()
    else:
        return -999


def compute_missingness(df, package):
    """Computes missingness of data with the provided package.

    Parameters
    ----------
    df : pd.DataFrame
        contains the data
    package : str
        visualisation package, options are:
        - "upset"
        - "pace"
    Returns
    -------
    pd.DataFrame
        containing the data missingness information
    """
    if package == "upset":
        return upsetplot.from_indicators(indicators=pd.isna, data=df)
    elif package == "pace":
        return Membership.from_data_frame(df)  # TODO: check this
    else:
        raise ValueError


def eval_data(df, package):
    """
    Evaluates the performance of UpSet by timing the
    missingness computation and the visualisation of the provided data.

    Parameters
    ----------
    df : pd.DataFrame
        data frame
    package : str
        visualisation package, options are:
        - "upset"
        - "pace"

    Returns
    -------
    results : float
        time [ms] how long it takes to compute missingness and visualise it

    """
    try:
        data_missing = compute_missingness(df, package)

        # TODO: how to time plot for pace?
        if package == "upset":
            upsetplot.plot(data_missing, show_counts=True)
            # plt.show(block=False)
            plt.savefig("tmp.pdf")

    except:
        raise
