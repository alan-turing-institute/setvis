import logging

from typing import Sequence, Callable, Optional, Any, List, Tuple

import numpy as np
import pandas as pd

from typing import Sequence, Callable, Optional, Any, List
from .setexpression import Set, SetExpr


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Col(Set):
    """An alias for :class:`~setvis.setexpression.Set`, used when referring
    to a 'column' (hence, Col) rather than a 'set' is more natural, for
    example, when referring to the set of elements with missing data in a
    particular column.
    """


def _invert_selection(universe: Sequence, selection: Optional[Sequence]):
    """Invert the given `selection` (with respect to the `universe` of possible values)

    A selection is either:

    - a sequence of distinct values from a 'universe' of possible values, or
    - the value `None`, meaning a selection containing the entire universe.

    :meta public:
    """

    if selection is None:
        return []
    elif len(selection) == 0:
        return None
    else:
        # equivalent to np.setdiff1d(universe, selection), but
        # setdiff1d doesn't preserve element order
        return list(np.array(universe)[~np.in1d(universe, selection)])


def selection_to_series(
    universe: Sequence, selection: Optional[Sequence], sort: bool = True
):
    """Convert a sequence of values (`selection`) into a boolean
    :class:`pd.Series` indexed by the universe of possible values
    (as given by `universe`).

    If `sort` is True (the default), the result has its index sorted.
    """
    result = pd.Series(np.in1d(universe, selection), index=universe)
    return result.sort_index() if sort else result


class Membership:
    """A storage-efficient representation of a collection of sets and the
    elements that belong to them (including their various
    intersections).  The representation is optimised for particular
    queries related to membership.

    An interpretation of sets as 'columns' and elements as
    boolean-valued 'records' is often useful, so sets/columns and
    elements/records are used interchangeably.

    Using one of the various named constructors
    (:func:`Membership.from_data_frame`, :func:`Membership.from_csv`,
    :func:`Membership.from_membership_data_frame` or
    :func:`Membership.from_membership_csv`) is the preferred way to
    construct a Membership object.

    If directly using the default constructor, the internal
    representation of the data is passed as two pandas dataframes,
    with these and the other arguments as given below.

    Parameters
    ----------
    intersection_id_to_columns: pd.DataFrame

        A dataframe with as many rows as there are unique
        intersections (patterns of membership or missingness), and a
        column for each set (or column in the original dataframe)

    intersection_id_to_records: pd.DataFrame

        A dataframe with as many rows as there are records.  The index
        of this DataFrame must be named "intersection_id" with a
        foreign key relationship to `intersection_id_to_columns`. It
        doesn't have to be a *unique* index (and generally won't be).
        This dataframe otherwise has a single column

    check: bool

        Check that the internal representation satisfies the required
        invariants.

    """

    _intersection_id_to_columns: pd.DataFrame
    _intersection_id_to_records: pd.DataFrame

    def __init__(
        self,
        intersection_id_to_columns: pd.DataFrame,
        intersection_id_to_records: pd.DataFrame,
        set_mode: bool = False,
        check: bool = True,
    ):
        self._intersection_id_to_columns = (
            intersection_id_to_columns.rename_axis("intersection_id")
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
        """The names of the sets included in the Membership object (or
        equivalently, the 'columns').
        """
        return list(self._intersection_id_to_columns)

    def intersections(self) -> pd.DataFrame:
        """Return all the distinct patterns of set membership in this
        Membership object.

        Each element belongs to a unique intersection of, for each
        set, either the set itself or its complement.  For `N` sets,
        `2**N` such intersections are possible.  This function returns
        all intersections with at least one element.

        The value returned is a DataFrame, mapping an 'intersection
        id' to a boolean for each of the original sets, indicating
        whether an element must be included (True) or excluded (False)
        from the set to be included in the intersection.
        """
        return self._intersection_id_to_columns

    def records(self) -> np.ndarray:
        """An array containing all record ids"""
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
        """A helper function that returns a boolean-valued pandas Series
        indicating the 'empty' intersection (`True` for this
        intersection_id, `False` for all others).

        The 'empty' intersection is the unique intersection excluding
        all of the original sets.
        """
        return ~self._intersection_id_to_columns.any(axis=1)

    def select_columns(self, selection: Optional[List] = None) -> "Membership":
        """Return a new Membership object for a subset of the columns

        A new Membership object is constructed from the given columns
        :param selection: (which must be a subset of columns of the
        current object).  Intersections that are identical under the
        selection are consolidated.

        It does not make sense to compare intersection ids before and
        after the selection (which relate to a different number of
        sets).
        """
        if selection is None:
            return self

        new_groups = self._intersection_id_to_columns[selection].groupby(
            selection, as_index=False
        )

        new_intersection_id_to_columns = new_groups.first()

        group_mapping = new_groups.ngroup().rename("intersection_id")

        new_intersection_id_to_records = (
            self._intersection_id_to_records.set_index(
                self._intersection_id_to_records.index.map(group_mapping)
            ).sort_index()
        )

        return self.__class__(
            new_intersection_id_to_columns,
            new_intersection_id_to_records,
            check=False,
        )

    def select_intersections(
        self, selection: Optional[Sequence] = None
    ) -> "Membership":
        """Return a Membership object with the given subset of intersections

        A Membership object is returned, based on the given intersections
        in :param selection: (which must be a subset of intersections of
        the current object).  A selection of `None` corresponds to every
        intersection being selected, and the original object is returned.

        The id of an intersection in the returned object is the same
        as in the original object before the selection was taken.
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

    def select_records(
        self, selection: Optional[Sequence[int]] = None
    ) -> "Membership":
        """Return a Membership object with the given subset of records

        A Membership object is returned, but only containing the given
        records in :param selection: (of course, these must be records
        present in the current object - it is an error to select a
        record that isn't present).  A selection of `None` corresponds
        to every record being selected, and the original object is
        returned.

        An 'intersection id' after the selection is made is consistent
        with the corresponding intersection before the selection.
        However, an intersection id before the selection may not be
        present after the selection (if there are no records in that
        particular intersection).
        """

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

    def drop_columns(self, selection: Optional[List]) -> "Membership":
        """Return a Membership object excluding the given column selection

        See also
        --------
        select_columns
        """
        return self.select_columns(self.invert_column_selection(selection))

    def drop_intersections(
        self, selection: Optional[Sequence[int]]
    ) -> "Membership":
        """Return a Membership object excluding the given intersection ids

        See also
        --------
        select_intersections
        """
        return self.select_intersections(
            self.invert_intersection_selection(selection)
        )

    def drop_records(self, selection: Optional[Sequence[int]]) -> "Membership":
        """Return a Membership object excluding the given record ids

        See also
        --------
        select_records
        """
        return self.select_records(self.invert_record_selection(selection))

    def invert_intersection_selection(
        self, selection: Optional[Sequence[int]]
    ) -> Optional[Sequence[int]]:
        """Invert a selection of intersection ids

        The result is a selection containing all intersection ids,
        except those contained in `selection`.
        """
        return _invert_selection(
            self._intersection_id_to_columns.index.values, selection
        )

    def invert_column_selection(
        self, selection: Optional[Sequence]
    ) -> Optional[Sequence]:
        """Invert a selection of column names

        The result is a selection containing all column names, except
        those contained in `selection`.
        """
        return _invert_selection(self.columns(), selection)

    def invert_record_selection(
        self, selection: Optional[Sequence[int]]
    ) -> Optional[Sequence[int]]:
        """Invert a selection of record ids

        The result is a selection containing all record ids, except
        those contained in `selection`.
        """
        return _invert_selection(self.records(), selection)

    def _compute_set(self, intersection_spec: SetExpr):
        """Evaluate the SetExpr :param:`intersection_spec` for the current instance"""
        return intersection_spec.evaluate(
            lookup_name=self._intersection_id_to_columns.__getitem__
        )

    def matching_intersections(self, intersection_spec: SetExpr) -> np.ndarray:
        """Return all intersections (by intersection id) that are included in
        `intersection_spec`
        """
        matches = self._compute_set(intersection_spec)

        # Convert Boolean array of matches to array of matching indices
        return matches.index[matches].values

    def matching_records(self, intersection_spec: SetExpr) -> np.ndarray:
        """Indicate which records are contained in matching intersections

        An intersection matches `intersection_spec` if the latter is a
        subset of it (or equal to it).

        Return a boolean series, which is True for each index where
        the data matches the given intersection.

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
        """Equivalent to ``len(self.matching_records(intersection_spec))``, but
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
        """Construct a Membership object from a dataframe

        This function produces a Membership object from a dataframe,
        interpretting the records according to one of two *modes*,
        'missingness mode' or 'set mode' (see below).

        In either case, the sets under consideration are named by the
        columns of the data frame, and the elements belonging to one
        or more of them indicated by the records in the dataframe,
        with each element named after the index in the dataframe.

        How a record is interpretted as belonging to a particular set
        differs between the two modes.

        In **missingness mode** (the default mode, enabled explicitly
        by passing ``set_mode=False``), a record belongs to the set
        named after a column, if and only if evaluating ``is_missing``
        on the entry in that column is truthy.  The sets in this
        mode are used to capture the missingness of data by column.
        The particular intersection a record belongs corresponds to
        the pattern of missingness (which columns contain missing
        data) in that record.

        Example
        -------

        As an example of using missingness mode, consider a dataframe
        ``df``, given by

        ======= ========== ==========
        (index) variety    length
        ======= ========== ==========
        0       NA         5.0
        1       "green"    NA
        2       NA         1.0
        3       "orange"   2.5
        ======= ========== ==========

        There are three missing values in this dataframe, indicated by NA.

        ``Membership.from_data_frame(df)`` is an object representing
        the missing data.  This is equivalent to the data in the
        following dataframe (although it isn't represented in quite
        this way):

        ======= ================ ===============
        (index) variety missing? length missing?
        ======= ================ ===============
        0       True             False
        1       False            True
        2       True             False
        3       False            False
        ======= ================ ===============

        The mapping from the values in the dataframe to booleans is
        controlled by the `is_missing` argument.

        We observe that:

        - The records 0 and 2 have the same missingness pattern (of ``(True, False)``)
        - Record 3 has no missing elements

        and so on.  The other methods on this class can be used to
        answer such queries efficiently.


        In **set mode** (enabled by passing ``set_mode=True``), the
        membership of a record to a set is indicated by the (boolean)
        value in the corresponding column.


        Note
        ----

        Set mode is in fact equivalent to passing ``is_missing =
        lambda r: r.astype(bool)`` in 'missingness mode' on a subset
        of columns whose names begin with 'category@'.

        Example
        -------

        As an example of using set mode, consider a dataframe ``df``, given by

        ======= ========== ========== ==========
        (index) category@A category@B category@C
        ======= ========== ========== ==========
        0       True       False      True
        1       True       False      False
        2       True       False      True
        3       False      False      False
        ======= ========== ========== ==========

        We can construct a Membership object from this dataframe as:

        ``Membership.from_data_frame(df, set_mode=True)``

        In this example, there are four records (labelled 0--3 by the index
        column), and three sets (A, B and C).

        We can observe the following:

        - Record 0 is an element of both A and C, but not B
        - Record 0 is therefore an element of the intersection of A, C and the complement of B
        - Record 2 belongs to the same 'intersection' as record 0
        - Set B doesn't contain any elements (it is empty)
        - Record 3 isn't a member of any of the sets.  It is member of
          the 'empty' intersection (the intersection of the complements of A, B and C)


        Parameters
        ----------
        df: pd.DataFrame
            The input dataframe
        is_missing: any Callable, that returns a boolean
            A predicate used in 'missingness mode', to determine if an
            element of the dataframe is missing.  The default is
            ``pd.isnull``
        set_mode: bool
            The dataframe is interpreted as set_mode (True) or missingness
            mode (False)

        See also
        --------
        from_membership_data_frame
          for loading membership information from a dataframe with a
          different format

        """
        if set_mode:
            columns = [col for col in df.columns if "category@" in col]
            grouped = df.astype(bool).groupby(columns)
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
    def from_csv(
        cls, filepath_or_buffer, read_csv_args=None, **kwargs
    ) -> "Membership":
        """Construct a Membership object from a csv file

        The data is first loading into a pandas dataframe and passed to
        :func:`~Membership.from_data_frame`.

        Parameters
        ----------

        filepath_or_buffer: file-like
          The file-like to load with :func:`pandas.read_csv`

        read_csv_args: Dict
          A dictionary of keyword arguments forwarded to :func:`pandas.read_csv`

        **kwargs
          All other arguments are forwarded to :func:`~Membership.from_data_frame`

        See also
        --------
        from_data_frame
          Load data from a data frame

        """

        if read_csv_args is None:
            read_csv_args = {}

        df = pd.read_csv(filepath_or_buffer, **read_csv_args)
        return cls.from_data_frame(df, **kwargs)

    @classmethod
    def from_membership_data_frame(
        cls,
        df,
        membership_column="set_membership",
        membership_separator="|",
        **kwargs,
    ) -> "Membership":
        """Construct a Membership object from a 'membership' dataframe

        A membership dataframe is any dataframe with a column
        indicating membership of named sets, in a particular format.

        The column is by default named 'set_membership' (but this can
        be specified with `membership_column`).  Other columns are
        ignored.

        An entry in the membership column must be a string, to be
        interpreted as the name of the sets that a record belongs
        to, separated by "|" (the default, changed with the
        `membership_separator` argument)

        Additional keyword arguments are forwarded to the class
        constructor.

        Example
        -------

        .. code-block:: python

           df = pd.DataFrame({
               "set_membership": ["A", "A,B,C", "B,C", ""]
           })

           df_membership = Membership.from_membership_data_frame(
               df, membership_seperator=","
           )


        After running the above, ``df`` is

        ======= ==============
        (index) set_membership
        ======= ==============
        0       "A"
        1       "A,B,C"
        2       "B,C"
        3
        ======= ==============

        and ``df_membership`` is an object representing three sets, A
        (containing records 0 and 1), B (containing 1 and 2) and C
        (also containing 1 and 2).  Record 3 is not a member of any of
        the sets.

        Note
        ----

        This constructor is careful never to produce a 'dense'
        dataframe, so can be useful when there are many sparse sets to
        avoid constructing a large intermediate dataframe.
        """

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
            # Type is boolean (not int)
            .astype(bool)
        )
        intersection_id_to_columns = mship_pivot.rename_axis("intersection_id")

        return cls(
            intersection_id_to_records=intersection_id_to_records,
            intersection_id_to_columns=intersection_id_to_columns,
            set_mode=True,
            **kwargs,
        )

    @classmethod
    def from_membership_csv(
        cls, filepath_or_buffer, read_csv_args=None, **kwargs
    ) -> "Membership":
        """Construct a Membership object from a 'membership' csv file

        The data is first loading into a pandas dataframe and passed to
        :func:`~Membership.from_membership_data_frame`.

        Parameters
        ----------

        filepath_or_buffer: file-like
          The file-like to load with :func:`pandas.read_csv`

        read_csv_args: Dict
          A dictionary of keyword arguments forwarded to :func:`pandas.read_csv`

        **kwargs
          All other keyword arguments are forwarded to
          :func:`~Membership.from_membership_data_frame`

        See also
        --------
        from_membership_data_frame
        """

        if read_csv_args is None:
            read_csv_args = {}

        df = pd.read_csv(filepath_or_buffer, **read_csv_args)
        return cls.from_membership_data_frame(df, **kwargs)

    @classmethod
    def from_postgres(
        cls,
        conn,
        relation: str,
        key: str,
        schema: Optional[str] = 'public',
    ) -> "Membership":
        """Construct a Membership object from a postgres database connection.

        Currently, membership is determined as with 'missingness mode'
        (see :func:`~Membership.from_dataframe`).  An entry in the
        table is considered missing if it is ``NULL``.

        Note
        ----

        This constructor requires psycopg2 to be installed.

        Note
        ----

        The database connection must have permission to create
        temporary tables.

        Parameters
        ----------
        conn: pyscopg2 connection object
            the database connection
        relation: str
            the name of the relation (table) in the database from which to load data
        key: str
            the name of the column to use as the 'record id' (must be a unique key)
        schema: Optional[str]
            the name of the schema to which the relation belongs
            (default of ``public``, meaning the public schema).

        """
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
                    query1,
                    conn,
                    index_col="intersection_id",
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
                    query2,
                    conn,
                    index_col="intersection_id",
                ).sort_index()

        return cls(
            intersection_id_to_records=intersection_id_to_records,
            intersection_id_to_columns=intersection_id_to_columns,
        )
