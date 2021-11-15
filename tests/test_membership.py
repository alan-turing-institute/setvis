import pytest
import numpy as np
import pandas as pd
from pace.membership import Membership, Col


def example_df():
    return pd.DataFrame(
        data={
            "a": [1, pd.NA, pd.NA, 4, 2, 1, pd.NA],
            "b": [1, pd.NA, 1, pd.NA, 1, pd.NA, 1],
        }
    )


def test_membership_construction():
    """Basic checks of constructing Membership"""

    r1 = pd.DataFrame(data={"intersection_id": [0, 1, 1, 2]}).set_index(
        "intersection_id"
    )
    c1 = pd.DataFrame(data={"intersection_id": [0, 1, 2]}).set_index(
        "intersection_id"
    )

    Membership(intersection_id_to_columns=c1, intersection_id_to_records=r1)

    r2 = pd.DataFrame(data={"intersection_id": [0, 1, 1, 3]}).set_index(
        "intersection_id"
    )
    c2 = pd.DataFrame(data={"intersection_id": [0, 1, 2]}).set_index(
        "intersection_id"
    )

    with pytest.raises(AssertionError):
        Membership(intersection_id_to_columns=c2, intersection_id_to_records=r2)


def test_membership_from_df():
    """
    Check construction of Membership from a pandas data frame, and
    that the original dataframe missingness can be reconstructed.
    """

    df = example_df()

    m = Membership.from_data_frame(df)

    assert (
        m._intersection_id_to_records.join(m._intersection_id_to_columns)
        .set_index("_record_id")
        .sort_index()
        .equals(df.isnull())
    )


def test_membership_match():
    df = example_df()
    m = Membership.from_data_frame(df)

    expected1 = np.array([1, 2, 6])

    assert (np.sort(m.matching_records(Col("a"))) == expected1).all()

    expected2 = np.array([1, 2, 3, 5, 6])

    assert (np.sort(m.matching_records(Col("a") | Col("b"))) == expected2).all()
