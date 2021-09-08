import numpy as np
import pandas as pd
from typing import Sequence, Callable, Optional, Any, List
from .setexpression import Set, SetExpr

# alias for Set
class Col(Set):
    pass


class Missingness(object):

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

    def _pattern_selection_all(self) -> np.ndarray:
        return np.ones(self._pattern.shape[0], dtype=bool)

    def counts(self, count_col_name: str = "_count",) -> pd.DataFrame:
        """Return the missingness patterns of the data, and the count of each

        :param count_col_name: The name of the column in the result
        holding the count data

        :return: A dataframe containing the missingness patterns and
        the number of times each appears in the dataset

        """

        counts = self._missingness.index.value_counts().rename(count_col_name)

        return self._pattern.join(counts)

    def select_patterns(self, pattern_selection: Optional[Sequence] = None):
        """Return a new Missingness object for a subset of the patterns

        A new Missingness object is constructed from the given patterns
        :param pattern_selection: (which must be a subset of patterns of
        the current object). 

        """
        if pattern_selection is None:
            pattern_selection = self._pattern_selection_all()

        new_pattern = self._pattern.loc[pattern_selection]
        new_missingness = self._missingness.loc[pattern_selection].sort_index()

        return self.__class__(
            new_pattern,
            new_missingness,
            check=True,
            pattern_key=self._pattern_key,
            index_col_label=self._index_col_label,
        )

    def _compute_set(self, pattern_spec: SetExpr):
        """Evaluate the SetExpr :pattern_spec: for the current instance
        """

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

        return np.sum(
            len(self._missingness.loc[k]) for k in matching_pattern_keys
        )

    def select_columns(self, col_selection):
        """Return a new Missingness object for a subset of the columns

        A new Missingness object is constructed from the given columns
        :param col_selection: (which must be a subset of columns of
        the current object).  Patterns that are identical under the
        selection are consolidated.

        """
        new_groups = self._pattern[col_selection].groupby(
            col_selection, as_index=False
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


def heatmap_data(m: Missingness, count_col="_count", pattern_selection=None):
    counts = m.counts(count_col, pattern_selection)
    return (
        counts.astype(int)
        .mul(counts[count_col], axis=0)
        .drop(count_col, axis=1)
    )


def value_bar_chart_data(m: Missingness, pattern_selection=None):
    labels = m.column_labels()
    return pd.DataFrame(
        (
            m.matches(Col(label), pattern_selection).count()["_index"]
            for label in labels
        ),
        index=labels,
        columns=["_count"],
    )