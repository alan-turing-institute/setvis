import numpy as np
import pandas as pd
from typing import Sequence, Callable, Optional, Any, List
from .setexpression import Set, SetExpr

# alias for Set
class Col(Set):
    pass


def _invert_selection(universe, selection):
    if selection is None:
        return np.array([])
    elif len(selection) == 0:
        return None
    else:
        # equivalent to np.setdiff1d(universe, selection), but
        # setdiff1d doesn't preserve element order
        return np.array(universe)[~np.in1d(universe, selection)]


class Missingness:

    _pattern: pd.DataFrame
    _missingness: pd.DataFrame

    def __init__(
        self,
        pattern: pd.DataFrame,
        missingness: pd.DataFrame,
        check: bool = True,
        pattern_key: str = "pattern_key",
        index_col_label: str = "_index",
    ):
        self._pattern = pattern.rename_axis(pattern_key)
        self._missingness = missingness

        self._pattern_key = pattern_key
        self._index_col_label = index_col_label

        if check:
            self._check()

    def _check(self):
        """Validate the class data

        In particular: Check that the DataFrame '_missingness' has a
        column 'pattern_key' with a foreign key relationship to
        '_pattern'.

        """
        assert self._pattern_key == self._missingness.index.name
        assert self._missingness.index.isin(self._pattern.index).all()

    def column_labels(self) -> List[Any]:
        return list(self._pattern)

    def patterns(self):
        return self._pattern

    def record_indices(self):
        return self._missingness[self._index_col_label].values

    def counts(self, count_col_name: str = "_count") -> pd.DataFrame:
        """Return the missingness patterns of the data, and the count of each

        :param count_col_name: The name of the column in the result
        holding the count data

        :return: A dataframe containing the missingness patterns and
        the number of times each appears in the dataset

        """

        counts = self._missingness.index.value_counts().rename(count_col_name)

        return self._pattern.join(counts)

    def select_patterns(self, selection: Optional[Sequence] = None):
        """Return a Missingness object with the given subset of patterns

        A Missingness object is returned, based on the given patterns
        in :param selection: (which must be a subset of patterns of
        the current object).  A selection of None corresponds to every
        pattern being selected, and the original object is returned.

        """
        if selection is None:
            return self

        new_pattern = self._pattern.loc[selection]
        new_missingness = self._missingness.loc[selection].sort_index()

        return self.__class__(
            new_pattern,
            new_missingness,
            check=True,
            pattern_key=self._pattern_key,
            index_col_label=self._index_col_label,
        )

    def drop_patterns(self, selection: Optional[Sequence[int]]):
        if selection and len(selection) == 0:
            return self
        else:
            return self.select_patterns(
                self.invert_pattern_selection(selection)
            )

    def invert_pattern_selection(self, selection):
        return _invert_selection(self._pattern.index.values, selection)

    def invert_record_selection(self, selection):
        return _invert_selection(self.record_indices(), selection)

    def invert_column_selection(self, selection):
        s = _invert_selection(self.column_labels(), selection)
        return s if s is None else list(s)

    def _compute_set(self, pattern_spec: SetExpr):
        """Evaluate the SetExpr :pattern_spec: for the current instance"""
        return pattern_spec.run_with(self._pattern.__getitem__)

    def matches(self, pattern_spec: SetExpr) -> np.ndarray:
        """Indicate which records match the given missingness pattern

        Return a boolean series, which is True for each index where
        the data matches the given missingness pattern :param
        pattern_spec:

        """

        pattern_matches = self._compute_set(pattern_spec)

        # Convert Boolean array of matches to series of matching indices
        matching_pattern_keys = pattern_matches.index[pattern_matches]

        # np.concatenate needs at least one array, so handle empty case
        # separately
        if matching_pattern_keys.empty:
            return np.array([], dtype=np.int64)
        else:
            return np.concatenate(
                [
                    # performance of indexing with a list using loc is
                    # poor when the df has a non-unique index:
                    # instead, apply loc to each separate index and
                    # concatenate
                    self._missingness.loc[k].to_numpy().reshape(-1)
                    for k in matching_pattern_keys
                ],
            )

    def count_matches(self, pattern_spec: SetExpr) -> int:
        """Equivalent to len(self.matches(pattern_spec)), but could have a
        performance benefit
        """

        pattern_matches = self._compute_set(pattern_spec)

        # Convert Boolean array of matches to series of matching indices
        matching_pattern_keys = pattern_matches.index[pattern_matches]

        return sum(
            len(self._missingness.loc[k]) for k in matching_pattern_keys
        )

    def select_columns(self, selection: Optional[List] = None):
        """Return a new Missingness object for a subset of the columns

        A new Missingness object is constructed from the given columns
        :param selection: (which must be a subset of columns of the
        current object).  Patterns that are identical under the
        selection are consolidated.

        """
        if selection is None:
            return self

        new_groups = self._pattern[selection].groupby(
            selection, as_index=False
        )

        new_pattern = new_groups.first()

        group_mapping = new_groups.ngroup().rename(self._pattern_key)

        new_missingness = self._missingness.set_index(
            self._missingness.index.map(group_mapping)
        ).sort_index()

        return self.__class__(
            new_pattern,
            new_missingness,
            check=False,
            pattern_key=self._pattern_key,
            index_col_label=self._index_col_label,
        )

    def drop_columns(self, selection: Optional[List]):
        if selection and len(selection) == 0:
            return self
        else:
            return self.select_columns(self.invert_column_selection(selection))

    def select_records(self, selection: Optional[Sequence[int]] = None):
        if selection is None:
            return self

        # no need to sort_index, since selection is returned in index order
        new_missingness = self._missingness[
            self._missingness[self._index_col_label].isin(selection)
        ]

        # discard unused pattern indices, but do not reindex
        new_pattern = self._pattern.loc[new_missingness.index.unique()]

        return self.__class__(
            new_pattern,
            new_missingness,
            check=True,
            pattern_key=self._pattern_key,
            index_col_label=self._index_col_label,
        )

    def drop_records(self, selection: Optional[Sequence[int]]):
        if selection and len(selection) == 0:
            return self
        else:
            return self.select_records(self.invert_record_selection(selection))

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        pattern_key: str = "pattern_key",
        index_col_label: str = "_index",
        is_missing: Callable[[Any], bool] = pd.isnull,
    ):
        grouped = is_missing(df).groupby(list(df))
        missingness = (
            pd.DataFrame(grouped.ngroup(), columns=[pattern_key])
            .rename_axis(index_col_label)
            .reset_index()
            .set_index(pattern_key)
            .sort_index()
        )
        pattern = pd.DataFrame(grouped.indices.keys(), columns=df.columns)

        return cls(
            missingness=missingness, pattern=pattern, pattern_key=pattern_key
        )

    @classmethod
    def from_csv(cls, filepath_or_buffer, *args, **kwargs):
        df = pd.read_csv(filepath_or_buffer, *args, **kwargs)
        return cls.from_data_frame(df)

    @classmethod
    def from_postgres(cls, conn):
        raise NotImplementedError()


def heatmap_data(m: Missingness, count_col="_count"):
    counts = m.counts(count_col)
    return (
        counts.astype(int)
        .mul(counts[count_col], axis=0)
        .drop(count_col, axis=1)
    )


def value_bar_chart_data(m: Missingness):
    labels = m.column_labels()
    return pd.DataFrame(
        {"_count": [m.count_matches(Col(label)) for label in labels]},
        index=labels,
    )
