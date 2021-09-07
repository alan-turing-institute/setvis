import pytest
import pandas as pd
from pace.missingness import Missingness, Col


def example_df():
    return pd.DataFrame(
        data={
            "a": [1, pd.NA, pd.NA, 4, 2, 1, pd.NA],
            "b": [1, pd.NA, 1, pd.NA, 1, pd.NA, 1],
        }
    )


def test_missingness_construction():
    """Basic checks of constructing Missingness"""

    missingness1 = pd.DataFrame(data={"pattern_key": [0, 1, 1, 2]})
    pattern1 = pd.DataFrame(data={"a": [1, 2, 3]})

    Missingness(pattern=pattern1, missingness=missingness1)

    missingness2 = pd.DataFrame(data={"pattern_key": [0, 1, 1, 3]})
    pattern2 = pd.DataFrame(data={"a": [1, 2, 3]})

    with pytest.raises(AssertionError):
        Missingness(pattern=pattern2, missingness=missingness2)


def test_missingness_from_df():
    """
    Check construction of Missingness from a pandas data frame, and
    that the original dataframe missingness can be reconstructed.
    """

    df = example_df()

    m = Missingness.from_data_frame(df)
    assert (
        m._missingness.join(m._pattern, on="pattern_key")
        .drop("pattern_key", axis=1)
        .equals(df.isnull())
    )


def test_missingness_match():
    df = example_df()
    m = Missingness.from_data_frame(df)

    expected1 = pd.Series([False, True, True, False, False, False, True])

    assert m.matches(Col("a")).equals(expected1)

    expected2 = pd.Series([False, True, True, True, False, True, True])

    assert m.matches(Col("a") | Col("b")).equals(expected2)
