import pandas as pd
from typing import Sequence, Callable, Optional, Any
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
    ):
        self._pattern = pattern
        self._missingness = missingness

        self._pattern_key = pattern_key

        if check:
            self._check()

    def _check(self):
        """Validate the class data

        In particular: Check that the DataFrame '_missingness' has a
        column 'pattern_key' with a foreign key relationship to
        '_pattern'.

        """
        assert self._pattern_key in self._missingness.keys()
        assert (
            self._missingness[self._pattern_key]
            .isin(self._pattern.index)
            .all()
        )

    def counts(
        self,
        count_col_name: str = "_count",
        selection: Optional[Sequence] = None,
    ) -> pd.DataFrame:
        """Return the missingness patterns of the data, and the count of each

        :param count_col_name: The name of the column in the result
        holding the count data

        :return: A dataframe containing the missingness patterns and
        the number of times each appears in the dataset

        """

        selection1 = selection or self._missingness.index

        counts = pd.DataFrame(
            self._missingness.loc[selection1]
            .groupby(self._pattern_key)
            .size(),
            columns=[count_col_name],
        )

        return self._pattern.join(counts, how="right")

    def _run_set(self, pattern_spec):
        """Evaluate a SetExpr for the current instance"""

        return pattern_spec.run_with(self._pattern.__getitem__)

    def matches(
        self, pattern_spec: SetExpr, row_selection: Optional[Sequence] = None,
    ) -> pd.Series:
        """Indicate which records match the given missingness pattern

        Return a boolean series with indices from :param
        row_selection:, which is True for each index where the data
        matches the given missingness pattern :param pattern_spec:.

        """
        row_selection1 = row_selection or self._missingness.index
        selected = self._missingness.loc[row_selection1]

        return selected[self._pattern_key].isin(
            self._pattern[self._run_set(pattern_spec)].index
        )

    def select_columns(self, col_selection) -> pd.DataFrame:
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

        new_missingness = self._missingness.merge(
            group_mapping,
            left_on=self._pattern_key,
            right_index=True,
            suffixes=("_", None),
        ).drop(self._pattern_key + "_", axis=1)

        return Missingness(new_pattern, new_missingness)

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        pattern_key: str = "pattern_key",
        is_missing: Callable[[Any], bool] = pd.isnull,
    ):
        grouped = is_missing(df).groupby(list(df))
        return cls(
            missingness=pd.DataFrame(grouped.ngroup(), columns=[pattern_key]),
            pattern=pd.DataFrame(grouped.indices.keys(), columns=df.columns),
            pattern_key=pattern_key,
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
