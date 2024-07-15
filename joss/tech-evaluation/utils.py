import math

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import upsetplot
import collections.abc
from setvis.membership import Membership
from setvis.plots import PlotSession


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
        max_comb.append(math.comb(num_cols, l1 + 1))

        if l1 > 0 and l1 < num_cols - 1:
            # Don't calculate this for combination length = 1 or num_cols
            length_ncomb.append([l1 + 1, max_comb[-1]])

    # Sort the list of lists
    length_ncomb.sort(key=lambda x: x[1])

    if (
        num_combinations <= sum(max_comb)
        and num_rows >= num_combinations
        and num_combinations >= num_cols + (num_cols - 1)
    ):
        # Calculate the number of combinations of each length (1 column, 2 columns, etc.) so the distribution is as even as possible
        # If you're interested 'esi' is an acronym for exclusive set intersections
        esi_params = {1: num_cols, num_cols: 1}
        ncombs = esi_params[1] + esi_params[num_cols]
        # Loop over the other combination lengths, leaving the one that can have the most combinations until last
        cnt = 0
        for item in length_ncomb:
            clen = item[0]
            esi_params[clen] = min(
                item[1],
                int(
                    (num_combinations - ncombs + len(length_ncomb) - cnt - 1)
                    / (len(length_ncomb) - cnt)
                ),
            )
            ncombs = ncombs + esi_params[clen]
            cnt = cnt + 1

        intersections_template = np.ones(
            shape=(num_combinations, num_cols)
        )  # matrix of ones

        # Loop over the parameter dictionary to create a list of lists. Each
        # item is a combination of missing values
        ii = 0
        for key, value in esi_params.items():
            # Randomise the order of the sets for ESIs with this cardinality
            columns = np.arange(1, num_cols + 1)
            # The number of ESIs and transactions with this cardinality
            esi_count = 0
            # Loop until the specified number of ESIs has been generated
            for i in itertools.combinations(columns, key):
                for n in i:
                    intersections_template[ii][n - 1] = np.nan

                # print(intersections_template[ii, :])
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
        df = (
            pd.DataFrame(rows, columns=cols, dtype=object)
            .sample(frac=1)
            .reset_index(drop=True)
        )
    else:
        # num_rows or num_combinations is too small to generate a general missing data pattern for num_cols columns
        print(
            "ERROR",
            "num_rows =",
            num_rows,
            "num_cols =",
            num_cols,
            "num_combinations =",
            num_combinations,
            "min allowed =",
            num_cols * 2 - 1,
            "max_possible =",
            sum(max_comb),
        )
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
    intersections_template = np.ones(
        shape=(num_cols, num_cols)
    )  # matrix of 0,1 of size nr_int x nr_col
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
    return (
        pd.DataFrame(rows, columns=cols, dtype=object)
        .sample(frac=1)
        .reset_index(drop=True)
    )


def compute_missingness(df, package):
    """Computes missingness of data with the provided package.

    Parameters
    ----------
    df : pd.DataFrame
        contains the data
    package : str
        visualisation package, options are:
        - "upset"
        - "setvis"
    Returns
    -------
    pd.DataFrame
        containing the data missingness information
    """
    if package == "upset":
        return upsetplot.from_indicators(indicators=pd.isna, data=df)
    elif package == "setvis":
        return Membership.from_data_frame(df)  # TODO: check this
    else:
        raise ValueError


def get_dataframe_sets(df):
    """
    Convert a dataframe into a dictionary that defines the sets and their members.
    The dictionary is suitable for upsetplot.from_contents()

    Parameters
    ----------
    df : pd.DataFrame
        data frame

    Returns
    -------
    contents : dict
        Each key is a set and the values is a list of the records in that set.

    """

    contents = {}
    # Loop over the data frame's columns
    for col in df.columns:
        groups = df.groupby(by=col)
        # Loop over the distinct values in this column
        for name, group in groups:
            # The column/value is a set. Store the records in it.
            contents[col + "_" + str(name)] = group.index.tolist()

    return contents


def generate_set_data(num_rows, num_cols, num_unique_values):
    """
    This function generates a pandas DataFrame with num_rows number of rows and num_cols number of columns,
    where each column contains a certain number of unique values. The easiest way to make sure there is a
    specific number of intersections is for num_rows to be an exact multiple of each value in num_unique_values,
    and one column's num_unique_values to be an exact multiple of the other column's num_unique_values.
    E.g., if num_rows = 100, num_cols = 2 and num_unique_values = [10, 50] then:

    Each value of column 1 appears in 100 / 10 = 10 rows
    Each value of column 2 appears in 100 / 50 = 2 rows
    number of intersections = 50

    If you want to generate sets and no intersections then set num_cols = 1. The number of sets = num_unique values

    The values are randomly shuffled before returning the DataFrame

    Parameters
    ----------
    num_rows : int
        The number of rows for the dataframe.
    num_cols : int
        The number of columns for the dataframe.
    num_unique_values : list
        The number of unique values for each column of the dataframe.

    Returns
    -------
    Dataframe : pandas DataFrame
    The generated dataframe with the specified number of rows and columns containing the set of unique values, shuffled randomly.

    """

    # Validate the parameters
    if num_cols > 2:
        print('*** ERROR *** generate_set_data() num_cols must be 1 or 2')
        return None

    if len(num_unique_values) != num_cols:
        print('*** ERROR *** generate_set_data() num_unique_values must have one entry for each column')
        return None
    elif max(num_unique_values) > num_rows:
        print('*** ERROR *** generate_set_data() an entry in num_unique_values is greater than num_rows')
        return None

    cols = ["category@v" + str(i) for i in range(num_cols)]
    data = {}
    # Loop over the columns
    for i in range(num_cols):
        values = []
        nrows = 0

        for vv in range(num_unique_values[i]):
            # Calculate the number of rows that have this value
            num = int(
                (num_rows - nrows + num_unique_values[i] - vv - 1) / (num_unique_values[i] - vv))
            # Append the values
            values = values + [str(vv)] * num
            nrows = nrows + num

        data[cols[i]] = values

    # pd.DataFrame.from_dict(data).sample(frac=1).reset_index(drop=True)
    return (
        pd.DataFrame.from_dict(data).sample(frac=1).reset_index(drop=True)
    )


def generate_data(pattern, num_rows, num_cols, num_int):
    """Generate data with the given pattern.

    Parameters
    ----------
    pattern : str
        Options:
        - "planned missing"
        - "general missing"
        - "sets"
    num_rows : int
        The number of rows for the dataframe.
    num_cols : int
        The number of columns for the dataframe.
    num_int : int
        The number of unique values for each column of the dataframe.

    Returns
    -------
    df : pd.DataFrame
        Contains the generated data
    """
    if pattern == "planned missing":
        df = generate_planned_missing(num_rows, num_cols)
    elif pattern == "general missing":
        df = generate_general_missing(num_rows, num_cols, num_int)
    elif pattern == "sets":
        # print("HACK num_values")
        num_values = []
        if isinstance(num_int, collections.abc.Sequence):
            num_values = num_int
        else:
            num_values = [num_cols]
        df = generate_set_data(
            num_rows,
            len(num_values) if isinstance(
                num_int, collections.abc.Sequence) else 1,
            num_values)
    else:
        df = None
    return df


def compute(df, package, set_mode):
    """Computes the datastructure required for plotting by given package

    Parameters
    ----------
    df : pd.DataFrame
        contains the dataset for which we compute set members
    package : string
        upset, setvis

    Returns
    -------
    data : pd.DataFrame
        contains the data structure with the set membership (missingness) information used for plotting

    """
    if package == "upset":
        if set_mode:
            contents = get_dataframe_sets(df)
            data = upsetplot.from_contents(contents)
        else:
            data = upsetplot.from_indicators(indicators=pd.isna, data=df)
    else:
        if set_mode:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'element@index',
                      'category@v0': 'v0', 'category@v1': 'v1'}, inplace=True)
            df['set_membership'] = df.apply(
                lambda row: 'v0_' + row[df.columns[1]] + '|' + 'v1_' + row[df.columns[2]], axis=1)
            data = Membership.from_membership_data_frame(
                df[['element@index', 'set_membership']])
        else:
            data = Membership.from_data_frame(df, set_mode=set_mode)
    return data


def plot_data(df, data, package):
    """Plots the data with the given visualisation package.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to use with setvis
    data : pd.DataFrame
        The data from upsetplot to use with upsetplot
    package : str
        The name of the visualisation package to use. Supported values are
        "upset" and "setvis"
    """
    if package == "upset":
        upsetplot.plot(data, show_counts=True)
        plt.show()
    else:
        session = PlotSession(df)
        session.add_plot("myplot")
