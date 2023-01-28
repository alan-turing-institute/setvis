import pandas as pd
import numpy as np
from .setexpression import Set
import logging
from typing import Tuple  # Sequence, Callable, Optional, Any, List, Tuple
from .membership import Membership

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# alias for Set
class Col(Set):
    pass


def set_bar_chart_data(
    m: Membership,
    sort_x_by=None,
    sort_x_order=None,
) -> pd.DataFrame:
    """Returns data used to generate a SetBarChart plot.

    The function counts the number of records in each set (column) and the number
    of records that are member of the empty set. The empty set contains all records
    that are not member of any set.

    Parameters
    ----------
    m: Membership
        Membership object
    sort_x_by: str
        Name of the sort option for the x-axis.
        Sort options are:
        - "value": sorts the bars along the x-axis with ascending or descending
          y-value as specified in `sort_x_order`
        - "alphabetical": sorts the bars along the x-axis in alphabetical order
          as specified in `sort_x_order`
        - default: if none of the above is provided the bars are sorted
          in the order they appear in the dataset.
    sort_x_order: str
        - "ascending" (default)
        - "descending"

    Returns
    -------
    pd.DataFrame
        DataFrame that contains the count of members of each set.

    """

    labels = m.columns()
    sets = [m.count_matching_records(Col(label)) for label in labels]
    if m._set_mode:
        empty_set = m.empty_intersection()
        labels += ["empty"]
        sets += [
            m.count_intersections()[empty_set]["_count"]
            .reset_index(drop=True)
            .get(0, 0)
        ]

    data = pd.DataFrame(
        {"_count": sets},
        index=labels,
    )

    # sort data if required by user
    if sort_x_order == "descending":
        ascending = False
    else:
        ascending = True
    if sort_x_by == "alphabetical":
        data = data.sort_index(ascending=ascending)
    elif sort_x_by == "value":
        data = data.sort_values(by="_count", ascending=ascending)

    return data


def intersection_heatmap_data(
    m: Membership,
    sort_x_by=None,
    sort_y_by=None,
    sort_x_order=None,
    sort_y_order=None,
) -> pd.DataFrame:
    """Returns data used to generate an IntersectionHeatmap plot.

    Creates a matrix with dimensions unique set intersections x columns
    (including the empty set).
    The values of the matrix are determined as follows:
    For each matrix row (set intersection), the value is
    - given by the count of records with this particular intersection if the column is
      part of the set intersection
    - 0 otherwise


    Parameters
    ----------
    m: Membership
        Membership object on which the plots are based.
    sort_x_by: str
        Name of the sort option for the x-axis.
        Sort options are:
        - "alphabetical": sorts the fields along the x-axis in alphabetical order
          as specified in `sort_x_order`
        - default: if none of the above is provided the fields on the x-axis
          of the heatmap are sorted in the order they appear in the dataset.
    sort_y_by: str
        Name of the sort option for the y-axis.
        Sort options are:
        - "value": sorts the fields along the y-axis by the heatmap value with
          the order as specified in `sort_x_order`
        - "length": sorts the fields along the y-axis by the intersection length
          with the order as specified in `sort_x_order`
        - default: if none of the above is provided the intersections on the y-axis
          of the heatmap are sorted in the order they appear in the dataset.
    sort_x_order: str
        - "ascending" (default)
        - "descending"
    sort_y_order: str
        - "ascending" (default)
        - "descending"

    Returns
    -------
    pd.DataFrame
        Contains the heatmap matrix.

    """
    counts = m.count_intersections().copy()  # don't modify original df
    if m._set_mode:
        counts["empty"] = m.empty_intersection()

    data = counts.astype(int).mul(counts["_count"], axis=0)

    # sort data if required by user
    if sort_x_order == "descending":
        reverse = True
    else:
        reverse = False

    if sort_y_order == "descending":
        ascending = False
    else:
        ascending = True

    if sort_y_by == "value":
        data = data.sort_values(by="_count", ascending=ascending)
    data = data.drop("_count", axis=1)
    if sort_y_by == "length":  # sort by combination length
        data["_len"] = np.count_nonzero(m.intersections(), axis=1)
        data = data.sort_values(by="_len", ascending=ascending).drop(
            ["_len"], axis=1
        )
    if sort_x_by == "alphabetical":
        data = data.reindex(sorted(data.columns, reverse=reverse), axis=1)
    return data


def set_cardinality_histogram_data(
    m: Membership,
    bins: int = 11,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.array]:
    """Returns data used to generate an SetCardinalityHistogram plot.

    Computes the histogram bins over the number of records that are
    member of each set. This includes the empty set.

    Parameters
    ----------
    m: Membership
        Membership object
    bins: int, optional
        Number of bins

    Returns
    -------
    data: pd.DataFrame
        DataFrame that contains the count of set members.
    column_data_source: pd.DataFrame
        DataFrame that contains the count for each bin
    edges: np.array
        Contains the bin edges which are used t

    """

    data = set_bar_chart_data(m)
    data_subset = data[data["_count"] != 0]
    _, edges = np.histogram(data_subset, bins=bins - 1)
    bin_ids = np.fmin(np.digitize(data_subset, edges), bins - 1)
    bin_ids += 1
    data["_bin_id"] = 1
    data["_bin_id"].loc[data_subset.index] = bin_ids[:, 0]

    # dict_data is for plotting
    keys = [x + 1 for x in range(bins)]
    vals = [data[data["_bin_id"] == x].shape[0] for x in keys]
    column_data_source = pd.DataFrame(
        {
            "_bin_id": keys,
            "_bin_count": vals,
        }
    )
    return data, column_data_source, edges


def intersection_cardinality_histogram_data(
    m: Membership,
    bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.array, int]:
    """Returns data used to generate an IntersectionCardinalityHistogram plot.

    Computes the histogram bins over the number of records with a
    given set interction.

    Parameters
    ----------
    m: Membership
        Membership object
    bins: int, optional
        Number of bins

    Returns
    -------
    data: pd.DataFrame
        DataFrame that contains the count of each unique set intersection.
    column_data_source: pd.DataFrame
        DataFrame that contains the count for each bin
    edges: np.array
        Contains the bin edges
    bins: int
        Number of bins. Can deviate from the input parameter bins
        if the number of unique values is less than the number of bins.

    """
    labels = list(m.count_intersections().index.values)
    data = pd.DataFrame(
        {"_count": m.count_intersections()["_count"]},
        index=labels,
    )
    bins = np.fmin(len(data["_count"].unique()), bins)
    data.index.name = "intersection_id"
    _, edges = np.histogram(data["_count"], bins)
    bin_ids = np.fmin(np.digitize(data["_count"], edges), bins)
    data["_bin_id"] = bin_ids - 1  # to match indices return by bokeh tools

    # dict for plotting
    list_bins = [x for x in range(bins)]
    count = [data[data["_bin_id"] == x].shape[0] for x in list_bins]
    column_data_source = pd.DataFrame({"_bin": list_bins, "_count": count})
    return data, column_data_source, edges, bins


def intersection_degree_histogram_data(
    m: Membership, bins: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, np.array, int]:
    """Returns data used to generate an IntersectionDegreeHistogram plot.

    Computes the histogram bins over the number of columns in
    each set intersection.

    Parameters
    ----------
    m: Membership
        Membership object
    bins: int, optional
        Number of bins

    Returns
    -------
    data: pd.DataFrame
        DataFrame that contains the number of fields in each unique
        set intersection.
    column_data_source: pd.DataFrame
        DataFrame that contains the count for each bin.
    edges: np.array
        Contains the bin edges.
    bins: int
        Number of bins. Can deviate from the input parameter bins
        if the number of unique values is less than the number of bins.

    """
    labels = list(m.intersections().index.values)
    data = pd.DataFrame(
        {"_length": list(m.intersections().sum(axis=1))},
        index=labels,
    )
    bins = np.fmin(len(data["_length"].unique()), bins)
    _, edges = np.histogram(data["_length"], bins)
    data["_bin_id"] = np.fmin(np.digitize(data["_length"], edges), bins) - 1

    # dict for plotting
    list_bins = [x for x in range(bins)]
    count = [data[data["_bin_id"] == x].shape[0] for x in list_bins]
    column_data_source = pd.DataFrame({"_bin": list_bins, "_count": count})
    return data, column_data_source, edges, bins


def intersection_bar_chart_data(
    m: Membership,
    sort_x_by: str = None,
    sort_x_order: str = None,
) -> pd.DataFrame:
    """Returns data used to generate an IntersectionBarChart plot.

    For every unique set intersection the function counts the number of records
    with this particular intersection.

    Parameters
    ----------
    m: Membership
        Membership object
    sort_x_by: str
        Name of the sort option for the x-axis.
        Sort options are:
        - "value": sorts the bars along the x-axis with ascending or descending
          y-value as specified in `sort_x_order`
        - "length": sorts the bars along the x-axis with ascending or descending
          intersection length as specified in `sort_x_order`
        - default: if none of the above is provided the bars are sorted
        with ascending y-values
    sort_x_order: str
        - "ascending" (default)
        - "descending"

    Returns
    -------
    pd.DataFrame
        DataFrame that contains the count of each set intersection.

    """
    data = m.count_intersections().copy()

    # sort axes
    if sort_x_order == "descending":
        ascending = False

    else:
        ascending = True

    if sort_x_by == "length":  # length of combination
        cols = m.intersections().columns
        data["_len"] = data.loc[:, cols].sum(axis=1)
        data = data.sort_values(by="_len", ascending=ascending)
    elif sort_x_by == "value":
        data = data.sort_values("_count", ascending=ascending)
    else:
        data = data.sort_values("_count", ascending=True)

    return data.reset_index()
