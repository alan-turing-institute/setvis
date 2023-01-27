import pytest
import pandas as pd
from pathlib import Path
import bokeh.plotting
from pandas.util.testing import assert_frame_equal

from pace.plots import (
    PlotBase,
    PlotSession,
    SetBarChart,
    SetCardinalityHistogram,
    IntersectionBarChart
)
from pace.membership import Membership
from pace.history import Selection


@pytest.fixture
def example_df():
    return pd.DataFrame(
        data={
            "a": [1, pd.NA, pd.NA, 4, 2, 1, pd.NA],
            "b": [1, pd.NA, 1, pd.NA, 1, pd.NA, 1],
        }
    )


@pytest.fixture
def simpsons_format1():
    test_source_path = Path(__file__).resolve()
    test_source_dir = test_source_path.parent
    dataset_dir = test_source_dir.parent / "examples" / "datasets"
    return pd.read_csv(dataset_dir / "simpsons - Format 1.csv")


def test_plotbase_construction(example_df):
    """
    Test that PlotBase raises a NotImplementedError when constructed directly.
    """
    with pytest.raises(NotImplementedError):
        PlotBase(
            Membership.from_data_frame(example_df),
            Selection()
        )


def test_PlotSession(simpsons_format1):
    """
    Test that a new plot can be added to a PlotSession and
    that it is added to the selection history.
    """
    set_session = PlotSession(simpsons_format1, set_mode=True)
    set_session.add_plot(name="a")
    assert ('a' in set_session.dict()['selection_history'])


def test_SetBarChart(simpsons_format1):
    """
    Test that a SetBarChart object can be created, that it produces
    a Bokeh figure, and that the plot_indices_to_selection and
    selection_to_plot_indices functions work correctly.
    And after use of the sample dataframe it remains
    the same as a copy made before.
    """
    copy = simpsons_format1.copy()
    set_bar = SetBarChart(
        Membership.from_data_frame(simpsons_format1, set_mode=True)
    )
    p = set_bar.plot()
    assert (isinstance(p, bokeh.plotting.Figure))
    seq = [1, 2]
    indi_to_sel = set_bar.plot_indices_to_selection(seq)
    sel_to_indi = set_bar.selection_to_plot_indices(indi_to_sel)
    assert (seq == sel_to_indi)
    assert_frame_equal(copy, simpsons_format1)


def test_SetCardinalityHistogram(simpsons_format1):
    """
    Test that a SetCardinalityHistogram object can be created,
    that it has the correct title and xlabel and
    that it produces a Bokeh figure.
    """
    card_hist = SetCardinalityHistogram(
        Membership.from_data_frame(simpsons_format1, set_mode=True)
    )
    assert (card_hist.title == "Set cardinality histogram")
    assert (card_hist.xlabel == "Cardinality of set")
    p = card_hist.plot()
    assert (isinstance(p, bokeh.plotting.Figure))


def test_IntersectionBarChart(simpsons_format1):
    """
    Test that a IntersectionBarChart object can be created, that it
    has the correct title and that it produces a Bokeh figure,
    and that the plot_indices_to_selection and selection_to_plot_indices
    functions work correctly.
    """
    card_hist = IntersectionBarChart(
        Membership.from_data_frame(simpsons_format1, set_mode=True)
    )
    assert (card_hist.title == "Intersection bar chart")
    p = card_hist.plot()
    assert (isinstance(p, bokeh.plotting.Figure))
    seq = [0, 1, 2]
    indi_to_sel = card_hist.plot_indices_to_selection(seq)
    # print(indi_to_sel)
    sel_to_indi = card_hist.selection_to_plot_indices(indi_to_sel)
    assert (seq == sel_to_indi)
