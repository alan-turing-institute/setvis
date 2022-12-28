from matplotlib.pyplot import axis
import numpy as np
import pandas as pd

from typing import Sequence, Callable, Optional, Any, List, Tuple
from .setexpression import Set, SetExpr
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# alias for Set
class Col(Set):
    pass


def _invert_selection(universe, selection):
    if selection is None:
        return []
    elif len(selection) == 0:
        return None
    else:
        # equivalent to np.setdiff1d(universe, selection), but
        # setdiff1d doesn't preserve element order
        return list(np.array(universe)[~np.in1d(universe, selection)])


def selection_to_series(universe, selection, sort=True):
    result = pd.Series(np.in1d(universe, selection), index=universe)
    return result.sort_index() if sort else result


class Membership:
    """Membership"""

    _intersection_id_to_columns: pd.DataFrame
    _intersection_id_to_records: pd.DataFrame

    def __init__(
        self,
        intersection_id_to_columns: pd.DataFrame,
        intersection_id_to_records: pd.DataFrame,
        set_mode: bool = False,
        check: bool = True,
    ):
        self._intersection_id_to_columns = intersection_id_to_columns.rename_axis(
            "intersection_id"
        )
        self._intersection_id_to_records = intersection_id_to_records
        self._set_mode = set_mode
        if check:
            self._check()

    def _check(self):
        """Validate the class data

        In particular: Check that the DataFrame
        '_intersection_id_to_records' has a column 'intersection_id'
        with a foreign key relationship to
        '_intersection_id_to_columns'.

        """
        assert self._intersection_id_to_records.index.name == "intersection_id"
        assert self._intersection_id_to_records.index.isin(
            self._intersection_id_to_columns.index
        ).all()

    def columns(self) -> List[Any]:
        return list(self._intersection_id_to_columns)

    def intersections(self):
        return self._intersection_id_to_columns

    def records(self):
        return self._intersection_id_to_records["_record_id"].values

    def count_intersections(self) -> pd.DataFrame:
        """Distinct set intersections, and the count of each

        :param count_col_name: The name of the column in the result
        holding the count data

        :return: A dataframe containing the intersections and the
        number of times each appears in the dataset

        """
        count_col_name = "_count"

        counts = self._intersection_id_to_records.index.value_counts().rename(
            count_col_name
        )

        return self._intersection_id_to_columns.join(counts)

    def empty_intersection(self) -> pd.Series:
        return ~self._intersection_id_to_columns.any(axis=1)

    def select_columns(self, selection: Optional[List] = None):
        """Return a new Membership object for a subset of the columns

        A new Membership object is constructed from the given columns
        :param selection: (which must be a subset of columns of the
        current object).  Intersections that are identical under the
        selection are consolidated.

        """
        if selection is None:
            return self

        new_groups = self._intersection_id_to_columns[selection].groupby(
            selection, as_index=False
        )

        new_intersection_id_to_columns = new_groups.first()

        group_mapping = new_groups.ngroup().rename("intersection_id")

        new_intersection_id_to_records = self._intersection_id_to_records.set_index(
            self._intersection_id_to_records.index.map(group_mapping)
        ).sort_index()

        return self.__class__(
            new_intersection_id_to_columns,
            new_intersection_id_to_records,
            check=False,
        )

    def select_intersections(self, selection: Optional[Sequence] = None):
        """Return a Membership object with the given subset of intersections

        A Membership object is returned, based on the given intersections
        in :param selection: (which must be a subset of intersections of
        the current object).  A selection of None corresponds to every
        intersection being selected, and the original object is returned.

        """
        if selection is None:
            return self

        new_intersection_id_to_columns = self._intersection_id_to_columns.loc[
            selection
        ]
        new_intersection_id_to_records = self._intersection_id_to_records.loc[
            selection
        ].sort_index()

        return self.__class__(
            new_intersection_id_to_columns,
            new_intersection_id_to_records,
            check=True,
        )

    def select_records(self, selection: Optional[Sequence[int]] = None):
        if selection is None:
            return self

        # no need to sort_index, since selection is returned in index order
        new_intersection_id_to_records = self._intersection_id_to_records[
            self._intersection_id_to_records["_record_id"].isin(selection)
        ]

        # discard unused intersection ids, but do not reindex
        new_intersection_id_to_columns = self._intersection_id_to_columns.loc[
            new_intersection_id_to_records.index.unique()
        ]

        return self.__class__(
            new_intersection_id_to_columns,
            new_intersection_id_to_records,
            check=True,
        )

    def drop_columns(self, selection: Optional[List]):
        return self.select_columns(self.invert_column_selection(selection))

    def drop_intersections(self, selection: Optional[Sequence[int]]):
        return self.select_intersections(
            self.invert_intersection_selection(selection)
        )

    def drop_records(self, selection: Optional[Sequence[int]]):
        return self.select_records(self.invert_record_selection(selection))

    def invert_intersection_selection(self, selection):
        return _invert_selection(
            self._intersection_id_to_columns.index.values, selection
        )

    def invert_column_selection(self, selection):
        return _invert_selection(self.columns(), selection)

    def invert_record_selection(self, selection):
        return _invert_selection(self.records(), selection)

    def _compute_set(self, intersection_spec: SetExpr):
        """Evaluate the SetExpr :intersection_spec: for the current instance"""
        return intersection_spec.run_with(
            self._intersection_id_to_columns.__getitem__
        )

    def matching_intersections(self, intersection_spec: SetExpr) -> np.ndarray:
        matches = self._compute_set(intersection_spec)
        if self._set_mode:
            matches = matches != 0

        # Convert Boolean array of matches to array of matching indices
        return matches.index[matches].values
        # return matches

    def matching_records(self, intersection_spec: SetExpr) -> np.ndarray:
        """Indicate which records match the given intersection

        Return a boolean series, which is True for each index where
        the data matches the given intersection :param
        intersection_spec:

        """
        matching_intersection_ids = self.matching_intersections(
            intersection_spec
        )

        # np.concatenate needs at least one array, so handle empty case
        # separately
        if len(matching_intersection_ids) == 0:
            return np.array([], dtype=np.int64)
        else:
            return np.concatenate(
                [
                    # performance of indexing with a list using loc is
                    # poor when the df has a non-unique index.
                    # Instead, apply loc to each separate index and
                    # concatenate
                    self._intersection_id_to_records.loc[k]
                    .to_numpy()
                    .reshape(-1)
                    for k in matching_intersection_ids
                ],
            )

    def count_matching_records(self, intersection_spec: SetExpr) -> int:
        """Equivalent to len(self.matching_records(intersection_spec)), but
        could have a performance benefit

        """
        matching_intersection_ids = self.matching_intersections(
            intersection_spec
        )

        return sum(
            len(self._intersection_id_to_records.loc[k])
            for k in matching_intersection_ids
        )

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        is_missing: Callable[[Any], bool] = pd.isnull,
        set_mode: bool = False,
    ):
        if set_mode:
            columns = [col for col in df.columns if "category@" in col]
            grouped = df.groupby(columns)
        else:
            columns = df.columns
            grouped = is_missing(df).groupby(list(df))
        intersection_id_to_records = (
            pd.DataFrame(grouped.ngroup(), columns=["intersection_id"])
            .rename_axis("_record_id")
            .reset_index()
            .set_index("intersection_id")
            .sort_index()
        )
        intersection_id_to_columns = pd.DataFrame(
            grouped.indices.keys(), columns=columns
        )

        return cls(
            intersection_id_to_records=intersection_id_to_records,
            intersection_id_to_columns=intersection_id_to_columns,
            set_mode=set_mode,
        )

    @classmethod
    def from_csv(cls, filepath_or_buffer, *args, **kwargs):
        df = pd.read_csv(filepath_or_buffer, *args, **kwargs)
        return cls.from_data_frame(df)

    @classmethod
    def from_membership_data_frame(
        cls,
        df,
        membership_column="set_membership",
        membership_separator="|",
        **kwargs,
    ):
        # This constructor is careful not to produce a 'dense'
        # dataframe, instead making the membership information into a
        # `frozenset` and grouping by that.

        grouped = pd.DataFrame(
            df[membership_column]
            .fillna("")
            # `"".split("|")` is `[""]`, which has the convenient consequence of
            # making a "" label for records that aren't in any set which will
            # not get dropped later.
            .str.split(membership_separator)
            .apply(frozenset)
        ).groupby(membership_column)

        intersection_id_to_records = (
            pd.DataFrame(grouped.ngroup(), columns=["intersection_id"])
            .rename_axis("_record_id")
            .reset_index()
            .set_index("intersection_id")
            .sort_index()
        )

        mship = pd.DataFrame(
            pd.DataFrame(grouped.indices.keys())
            .melt(ignore_index=False)["value"]
            .dropna()
            .sort_index()
        )
        mship["member"] = 1.0
        mship_pivot = (
            mship.pivot(columns=["value"])
            .droplevel(0, axis=1)
            # remove column for the explicitly 'empty' label
            # (errors="ignore" so no failure if it does not exist)
            .drop("", axis=1, errors="ignore")
            .rename_axis(None, axis=1)
            # Prepend "category@" to the label to make the column labels
            # consistent with the constructor(s) from 'Format 1' csv files
            .rename(lambda x: "category@" + x, axis=1)
            .fillna(0.0)
            .astype(int)
        )
        intersection_id_to_columns = mship_pivot.rename_axis("intersection_id")

        return cls(
            intersection_id_to_records=intersection_id_to_records,
            intersection_id_to_columns=intersection_id_to_columns,
            set_mode=True,
        )

    @classmethod
    def from_membership_csv(cls, filepath_or_buffer, *args, **kwargs):
        df = pd.read_csv(filepath_or_buffer, **kwargs)
        return cls.from_membership_data_frame(df, *args, **kwargs)

    @classmethod
    def from_postgres(
        cls, conn, relation: str, key: str, schema: Optional[str] = None,
    ):
        import psycopg2
        import psycopg2.extensions
        from psycopg2 import sql

        # Start a transaction on the connection
        with conn:
            with conn.cursor() as curs:
                # Determine the column headers from an empty selection
                curs.execute(
                    sql.SQL(
                        "SELECT * FROM {schema}.{relation} LIMIT 0"
                    ).format(
                        schema=sql.Identifier(schema),
                        relation=sql.Identifier(relation),
                    )
                )
                column_names = [c.name for c in curs.description]

                assert key in column_names

                columns_excl_key = [c for c in column_names if c != key]

                # Make a temporary table containing the
                # (boolean-valued) missingness of each non-key column
                columns_missing = [
                    sql.SQL("{col} IS NULL AS {col}").format(
                        col=sql.Identifier(col)
                    )
                    for col in columns_excl_key
                ]
                curs.execute(
                    sql.SQL(
                        """
                        CREATE TEMPORARY TABLE temp_missing
                        ON COMMIT DROP
                        AS
                          SELECT {key}, {columns_missing}
                          FROM {schema}.{relation}
                        """
                    ).format(
                        key=sql.Identifier(key),
                        columns_missing=sql.SQL(", ").join(columns_missing),
                        schema=sql.Identifier(schema),
                        relation=sql.Identifier(relation),
                    )
                )

                # Make another temporary table where each record is
                # labelled with a number identifying its distinct
                # intersections (missingness combinations in this
                # context)
                curs.execute(
                    sql.SQL(
                        """
                        CREATE TEMPORARY TABLE temp_intersections
                        ON COMMIT DROP
                        AS
                          SELECT ROW_NUMBER() OVER (
                            ORDER BY ({columns_excl_key})
                          ) - 1
                          AS intersection_id, *
                          FROM (
                            SELECT {columns_excl_key}
                            FROM temp_missing
                            GROUP BY ({columns_excl_key})
                          ) AS s
                        """
                    ).format(
                        columns_excl_key=sql.SQL(",").join(
                            sql.Identifier(c) for c in columns_excl_key
                        )
                    )
                )

                # First query: All intersections
                query1 = sql.SQL(
                    """
                    SELECT * FROM temp_intersections
                    ORDER BY intersection_id
                    """
                )

                intersection_id_to_columns = pd.read_sql_query(
                    query1, conn, index_col="intersection_id",
                )

                # Second query: intersection id to record id
                query2 = sql.SQL(
                    """
                    SELECT
                      intersection_id, {key} AS _record_id
                    FROM
                      temp_missing
                    NATURAL JOIN
                      temp_intersections
                    """
                ).format(key=sql.Identifier(key))

                intersection_id_to_records = pd.read_sql_query(
                    query2, conn, index_col="intersection_id",
                ).sort_index()

        return cls(
            intersection_id_to_records=intersection_id_to_records,
            intersection_id_to_columns=intersection_id_to_columns,
        )


def intersection_bar_chart_data(
    m: Membership,
    sort_x_by: str = None,
    sort_x_order: str = None,
) -> pd.DataFrame:
    """Return data needed to generate an IntersectionBarChart plot

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
    list_bins = list(range(bins))
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
    list_bins = list(range(bins))
    count = [data[data["_bin_id"] == x].shape[0] for x in list_bins]
    column_data_source = pd.DataFrame({"_bin": list_bins, "_count": count})
    return data, column_data_source, edges, bins


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

    # if (sort_x_by == "value") & ((sort_x_order == "descending")):
    #     # sort bin ids
    #     map_keys = [x + 1 for x in range(bins)]
    #     map_vals = [x for x in reversed(map_keys)]
    #     dict_mapping = dict(zip(map_keys, map_vals))
    #     data["_bin_id"] = data["_bin_id"].map(dict_mapping)

    #     # sort edges for histogram x-tick labels
    #     edges = [x for x in reversed(edges)]
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
    reverse = sort_x_order == "descending"
    ascending = sort_y_order != "descending"

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
