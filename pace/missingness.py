import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Callable, Optional, Union, Any


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
        """
        Check that the DataFrame 'missingness' has a column
        'missingness_pattern' with a foreign key relationship to
        'pattern'.
        """
        assert self._pattern_key in self._missingness.keys()
        assert self._missingness[self._pattern_key].isin(self._pattern.index).all()

    def select_columns(self, col_selection: Sequence):
        """
        Return a new Missingness object based on the column selection. 
        Combine missingness patterns.
        """
        new_groups = self._pattern[col_selection].groupby(col_selection, as_index=False)
        new_pattern = new_groups.first()
        group_mapping = new_groups.ngroup().rename(self._pattern_key)

        new_missingness = self._missingness.merge(
            group_mapping,
            left_on=self._pattern_key,
            right_index=True,
            suffixes=("_", None),
        ).drop(self._pattern_key + "_", axis=1)

        return Missingness(new_pattern, new_missingness)

    def pattern_counts(
        self, count_col_name: str = "_count", selection: Optional[Sequence] = None
    ):
        """
        Return the missingness patterns of the data, along with the
        count of each in the data
        """
        selection1 = (selection or self._missingness.index,)

        counts = pd.DataFrame(
            self._missingness.loc[selection1].groupby(self._pattern_key).size(),
            columns=[count_col_name],
        )

        return self._pattern.join(counts, how="right")

    # def pattern_counts_alt(self, col_name: str = "_count"):
    #     """
    #     Alternative implementation of pattern_counts for performance comparison
    #     """
    #     groups = self._missingness.groupby(self._pattern_key).groups
    #     return self._pattern.join(
    #         pd.Series((len(v) for v in groups.values()), name=col_name)
    #     )

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
    counts = m.pattern_counts(count_col)
    return counts.astype(int).mul(counts[count_col], axis=0).drop(count_col, axis=1)


## TODO

# add a selection argument to everything

# separate class for keeping track of selection and history

# method for looking up based on various selections
