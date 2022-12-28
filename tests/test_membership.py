import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pace.membership import Membership, Col


@pytest.fixture
def example_df():
    return pd.DataFrame(
        data={
            "a": [1, pd.NA, pd.NA, 4, 2, 1, pd.NA],
            "b": [1, pd.NA, 1, pd.NA, 1, pd.NA, 1],
        }
    )


@pytest.fixture
def simpsons_format2():
    test_source_path = Path(__file__).resolve()
    test_source_dir = test_source_path.parent
    dataset_dir = test_source_dir.parent / "examples" / "datasets"

    return pd.read_csv(dataset_dir / "simpsons - Format 2.csv")


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


def test_membership_from_df(example_df):
    """
    Check construction of Membership from a pandas data frame, and
    that the original dataframe missingness can be reconstructed.
    """

    m = Membership.from_data_frame(example_df)

    assert (
        m._intersection_id_to_records.join(m._intersection_id_to_columns)
        .set_index("_record_id")
        .sort_index()
        .equals(example_df.isnull())
    )


def test_membership_set_mode_no_empty(simpsons_format2):
    """Check construction (no exceptions) of Membership in set mode from
    compressed format.

    Regression of GH #77: failure when no element is a member of none
    of the sets.
    """

    # First four rows have no 'empty' members
    df = simpsons_format2[0:4]

    # No exception
    Membership.from_membership_data_frame(df)


def test_membership_match(example_df):
    m = Membership.from_data_frame(example_df)

    expected1 = np.array([1, 2, 6])

    assert (np.sort(m.matching_records(Col("a"))) == expected1).all()

    expected2 = np.array([1, 2, 3, 5, 6])

    assert (np.sort(m.matching_records(Col("a") | Col("b"))) == expected2).all()
