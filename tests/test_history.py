import pytest
import numpy as np
import pandas as pd
from pandas import NA
from pace.missingness import Missingness
from pace.history import SelectionHistory, Selection


def example_df():
    return pd.DataFrame(
        data={
            "a": [1, NA, NA, 1, 1, NA, 1],
            "c": [1, NA, 1, NA, NA, 1, 1],
            "b": [1, NA, NA, NA, NA, NA, 1],
        }
    )


def test_selection_history():
    m = Missingness.from_data_frame(example_df())
    h = SelectionHistory(m)

    h.new_selection("a")
    h.update_active("a", exclude=Selection(columns=["a"]))

    assert h.missingness("a").column_labels() == ["c", "b"]
    assert len(h.missingness("a").patterns()) == 3

    h.new_selection("b", "a")
    h.update_active("b", exclude=Selection(patterns=[0, 1]))

    m_b = h.missingness("b")

    print(m_b.counts())

    # only one pattern remaining
    assert len(m_b.patterns()) == 1

    # three records left - those with the final pattern in sorted
    # order, '(True, True)'
    assert list(np.sort(m_b.record_indices())) == [1, 3, 4]

    # original missingness - check unchanged
    m_orig = h.missingness()
    assert m_orig.column_labels() == ["a", "c", "b"]
    assert (np.sort(m_orig.record_indices()) == np.arange(7)).all()
    assert (m_orig.patterns().index.values == np.arange(4)).all()
