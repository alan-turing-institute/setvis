import pytest
import numpy as np
import pandas as pd
from pandas import NA
from setvis.membership import Membership
from setvis.history import SelectionHistory, Selection


def example_df():
    return pd.DataFrame(
        data={
            "a": [1, NA, NA, 1, 1, NA, 1],
            "c": [1, NA, 1, NA, NA, 1, 1],
            "b": [1, NA, NA, NA, NA, NA, 1],
        }
    )


def test_selection_history():
    m = Membership.from_data_frame(example_df())
    h = SelectionHistory(m)

    h.new_selection("a")
    h["a"] = Selection(columns=["a"])

    assert h.membership("a").columns() == ["c", "b"]
    assert len(h.membership("a").intersections()) == 3

    h.new_selection("b", "a")
    h["b"] = Selection(intersections=[0, 1])

    m_b = h.membership("b")

    # only one pattern remaining
    assert len(m_b.intersections()) == 1

    # three records left - those with the final pattern in sorted
    # order, '(True, True)'
    assert list(np.sort(m_b.records())) == [1, 3, 4]

    # original membership - check unchanged
    m_orig = h.membership()
    assert m_orig.columns() == ["a", "c", "b"]
    assert (np.sort(m_orig.records()) == np.arange(7)).all()
    assert (m_orig.intersections().index.values == np.arange(4)).all()
