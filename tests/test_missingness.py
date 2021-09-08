import pytest
import numpy as np
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

    missingness1 = pd.DataFrame(data={"pattern_key": [0, 1, 1, 2]}).set_index(
        "pattern_key"
    )
    pattern1 = pd.DataFrame(data={"pattern_key": [0, 1, 2]}).set_index(
        "pattern_key"
    )

    Missingness(pattern=pattern1, missingness=missingness1)

    missingness2 = pd.DataFrame(data={"pattern_key": [0, 1, 1, 3]}).set_index(
        "pattern_key"
    )
    pattern2 = pd.DataFrame(data={"pattern_key": [0, 1, 2]}).set_index(
        "pattern_key"
    )

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
        m._missingness.join(m._pattern)
        .set_index("_index")
        .sort_index()
        .equals(df.isnull())
    )


def test_missingness_match():
    df = example_df()
    m = Missingness.from_data_frame(df)

    expected1 = np.array([1, 2, 6])

    assert (np.sort(m.matches(Col("a"))) == expected1).all()

    expected2 = np.array([1, 2, 3, 5, 6])

    assert (np.sort(m.matches(Col("a") | Col("b"))) == expected2).all()
