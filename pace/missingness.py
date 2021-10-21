import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extensions
from psycopg2 import sql

from typing import Sequence, Callable, Optional, Any, List
from .setexpression import Set, SetExpr

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
    result = pd.Series(np.in1d(universe, selection), index=universe,)
    return result.sort_index() if sort else result


class Missingness:

    _combination_id_to_columns: pd.DataFrame
    _combination_id_to_records: pd.DataFrame

    def __init__(
        self,
        combination_id_to_columns: pd.DataFrame,
        combination_id_to_records: pd.DataFrame,
        check: bool = True,
    ):
        self._combination_id_to_columns = combination_id_to_columns.rename_axis(
            "combination_id"
        )
        self._combination_id_to_records = combination_id_to_records

        if check:
            self._check()

    def _check(self):
        """Validate the class data

        In particular: Check that the DataFrame
        '_combination_id_to_records' has a column 'combination_id'
        with a foreign key relationship to
        '_combination_id_to_columns'.

        """
        assert self._combination_id_to_records.index.name == "combination_id"
        assert self._combination_id_to_records.index.isin(
            self._combination_id_to_columns.index
        ).all()

    def columns(self) -> List[Any]:
        return list(self._combination_id_to_columns)

    def combinations(self):
        return self._combination_id_to_columns

    def records(self):
        return self._combination_id_to_records["_record_id"].values

    def count_combinations(self) -> pd.DataFrame:
        """Distinct missingness combinations in the data, and the count of each

        :param count_col_name: The name of the column in the result
        holding the count data

        :return: A dataframe containing the missingness combinations and
        the number of times each appears in the dataset

        """
        count_col_name = "_count"

        counts = self._combination_id_to_records.index.value_counts().rename(
            count_col_name
        )

        return self._combination_id_to_columns.join(counts)

    def select_columns(self, selection: Optional[List] = None):
        """Return a new Missingness object for a subset of the columns

        A new Missingness object is constructed from the given columns
        :param selection: (which must be a subset of columns of the
        current object).  Combinations that are identical under the
        selection are consolidated.

        """
        if selection is None:
            return self

        new_groups = self._combination_id_to_columns[selection].groupby(
            selection, as_index=False
        )

        new_combination_id_to_columns = new_groups.first()

        group_mapping = new_groups.ngroup().rename("combination_id")

        new_combination_id_to_records = self._combination_id_to_records.set_index(
            self._combination_id_to_records.index.map(group_mapping)
        ).sort_index()

        return self.__class__(
            new_combination_id_to_columns,
            new_combination_id_to_records,
            check=False,
        )

    def select_combinations(self, selection: Optional[Sequence] = None):
        """Return a Missingness object with the given subset of combinations

        A Missingness object is returned, based on the given combinations
        in :param selection: (which must be a subset of combinations of
        the current object).  A selection of None corresponds to every
        combination being selected, and the original object is returned.

        """
        if selection is None:
            return self

        new_combination_id_to_columns = self._combination_id_to_columns.loc[
            selection
        ]
        new_combination_id_to_records = self._combination_id_to_records.loc[
            selection
        ].sort_index()

        return self.__class__(
            new_combination_id_to_columns,
            new_combination_id_to_records,
            check=True,
        )

    def select_records(self, selection: Optional[Sequence[int]] = None):
        if selection is None:
            return self

        # no need to sort_index, since selection is returned in index order
        new_combination_id_to_records = self._combination_id_to_records[
            self._combination_id_to_records["_record_id"].isin(selection)
        ]

        # discard unused combination ids, but do not reindex
        new_combination_id_to_columns = self._combination_id_to_columns.loc[
            new_combination_id_to_records.index.unique()
        ]

        return self.__class__(
            new_combination_id_to_columns,
            new_combination_id_to_records,
            check=True,
        )

    def drop_columns(self, selection: Optional[List]):
        return self.select_columns(self.invert_column_selection(selection))

    def drop_combinations(self, selection: Optional[Sequence[int]]):
        return self.select_combinations(
            self.invert_combination_selection(selection)
        )

    def drop_records(self, selection: Optional[Sequence[int]]):
        return self.select_records(self.invert_record_selection(selection))

    def invert_combination_selection(self, selection):
        return _invert_selection(
            self._combination_id_to_columns.index.values, selection
        )

    def invert_column_selection(self, selection):
        return _invert_selection(self.columns(), selection)

    def invert_record_selection(self, selection):
        return _invert_selection(self.records(), selection)

    def _compute_set(self, combination_spec: SetExpr):
        """Evaluate the SetExpr :combination_spec: for the current instance"""
        return combination_spec.run_with(
            self._combination_id_to_columns.__getitem__
        )

    def matching_combinations(self, combination_spec: SetExpr) -> np.ndarray:
        matches = self._compute_set(combination_spec)

        # Convert Boolean array of matches to array of matching indices
        return matches.index[matches].values

    def matching_records(self, combination_spec: SetExpr) -> np.ndarray:
        """Indicate which records match the given missingness combination

        Return a boolean series, which is True for each index where
        the data matches the given missingness combination :param combination_spec:

        """
        matching_combination_ids = self.matching_combinations(combination_spec)

        # np.concatenate needs at least one array, so handle empty case
        # separately
        if len(matching_combination_ids) == 0:
            return np.array([], dtype=np.int64)
        else:
            return np.concatenate(
                [
                    # performance of indexing with a list using loc is
                    # poor when the df has a non-unique index.
                    # Instead, apply loc to each separate index and
                    # concatenate
                    self._combination_id_to_records.loc[k]
                    .to_numpy()
                    .reshape(-1)
                    for k in matching_combination_ids
                ],
            )

    def count_matching_records(self, combination_spec: SetExpr) -> int:
        """Equivalent to len(self.matching_records(combination_spec)), but
        could have a performance benefit

        """
        matching_combination_ids = self.matching_combinations(combination_spec)

        return sum(
            len(self._combination_id_to_records.loc[k])
            for k in matching_combination_ids
        )

    @classmethod
    def from_data_frame(
        cls, df: pd.DataFrame, is_missing: Callable[[Any], bool] = pd.isnull,
    ):
        grouped = is_missing(df).groupby(list(df))
        combination_id_to_records = (
            pd.DataFrame(grouped.ngroup(), columns=["combination_id"])
            .rename_axis("_record_id")
            .reset_index()
            .set_index("combination_id")
            .sort_index()
        )
        combination_id_to_columns = pd.DataFrame(
            grouped.indices.keys(), columns=df.columns
        )

        return cls(
            combination_id_to_records=combination_id_to_records,
            combination_id_to_columns=combination_id_to_columns,
        )

    @classmethod
    def from_csv(cls, filepath_or_buffer, *args, **kwargs):
        df = pd.read_csv(filepath_or_buffer, *args, **kwargs)
        return cls.from_data_frame(df)

    @classmethod
    def from_postgres(
        cls,
        conn: psycopg2.extensions.connection,
        relation: str,
        key: str,
        schema: Optional[str] = None,
    ):
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
                # labelled with a number identifying its missingness
                # combination
                curs.execute(
                    sql.SQL(
                        """
                        CREATE TEMPORARY TABLE temp_combinations
                        ON COMMIT DROP
                        AS
                          SELECT ROW_NUMBER() OVER (
                            ORDER BY ({columns_excl_key})
                          ) - 1
                          AS combination_id, *
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

                # First query: All missingness combinations
                combination_id_to_columns = (
                    pd.read_sql("SELECT * FROM temp_combinations", conn)
                    .set_index("combination_id")
                    .sort_index()
                )

                # Second query: combination id to record id
                query2 = sql.SQL(
                    """
                    SELECT
                      combination_id, {key}
                    FROM
                      temp_missing
                    NATURAL JOIN
                      temp_combinations
                    """
                ).format(key=sql.Identifier(key))

                combination_id_to_records = (
                    pd.read_sql(query2, conn)
                    .set_index("combination_id")
                    .sort_index()
                    .rename(columns={key: "_record_id"})
                )

        return cls(
            combination_id_to_records=combination_id_to_records,
            combination_id_to_columns=combination_id_to_columns,
        )


def heatmap_data(m: Missingness):
    counts = m.count_combinations()
    return (
        counts.astype(int).mul(counts["_count"], axis=0).drop("_count", axis=1)
    )


def value_bar_chart_data(m: Missingness):
    labels = m.columns()
    return pd.DataFrame(
        {"_count": [m.count_matching_records(Col(label)) for label in labels]},
        index=labels,
    )


def value_count_histogram_data(m: Missingness, bins: int = 10):
    labels = m.columns()
    data = pd.DataFrame(
        {"_count": [m.count_matching_records(Col(label)) for label in labels]},
        index=labels,
    )
    data["_bin_id"] = 1
    data_subset = data[
        data["_count"] != 0
    ]  # TODO: problem when hist based on selection
    if not data["_count"].min():  # if there are fields that are never missing
        # bins = bins - 1
        _, hist_edges = np.histogram(
            data_subset, bins=bins - 1
        )  # is this properly implemented??
        bin_ids = np.fmin(np.digitize(data_subset, hist_edges), bins - 1)
        bin_ids = bin_ids + 1
    else:
        _, hist_edges = np.histogram(data_subset, bins=bins)
        bin_ids = np.fmin(np.digitize(data_subset, hist_edges), bins)
    data["_bin_id"].loc[data_subset.index] = bin_ids[:, 0]

    # dict_data is for plotting
    keys = [x + 1 for x in range(bins)]
    vals = [data[data["_bin_id"] == x].shape[0] for x in keys]
    column_data_source = pd.DataFrame({"_bin_id": keys, "_bin_count": vals,})
    return data, column_data_source, hist_edges


def combination_bar_chart_data(m: Missingness):
    # sort with decreasign order
    return (
        m.count_combinations()
        .sort_values("_count", ascending=False)
        .reset_index()
    )


def combination_count_histogram_data(m: Missingness, bins: int = 10):
    labels = list(m.count_combinations().index.values)
    data = pd.DataFrame(
        {"_count": m.count_combinations()["_count"]}, index=labels,
    )
    data.index.name = "combination_id"
    _, edges = np.histogram(data["_count"], bins)
    bin_ids = np.fmin(np.digitize(data["_count"], edges), bins)

    data["_bin_id"] = bin_ids - 1

    # dict for plotting
    bins = [x for x in range(bins)]
    count = [data[data["_bin_id"] == x].shape[0] for x in bins]
    column_data_source = pd.DataFrame({"_bin": bins, "_count": count})
    return data, column_data_source, edges


def combination_length_histogram_data(m: Missingness, bins: int = 10):
    labels = list(m.combinations().index.values)
    data = pd.DataFrame(
        {"_length": list(m.combinations().sum(axis=1))}, index=labels,
    )
    _, edges = np.histogram(data["_length"], bins)
    bin_ids = np.fmin(np.digitize(data["_length"], edges), bins)
    data["_bin_id"] = bin_ids - 1

    # dict for plotting
    bins = [x for x in range(bins)]
    count = [data[data["_bin_id"] == x].shape[0] for x in bins]
    column_data_source = pd.DataFrame({"_bin": bins, "_count": count})
    return data, column_data_source, edges
