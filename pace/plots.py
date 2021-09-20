from pace.missingness import Missingness
import pace.missingness as missingness
from pace.history import SelectionHistory, Selection
import bokeh.plotting
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LinearColorMapper, tools, CustomJS
from bokeh.palettes import Oranges256
from bokeh.transform import transform, linear_cmap
from bokeh.models.widgets import Panel, Tabs  # not sure if needed
import bokeh.io
import pandas as pd
import numpy as np
import logging
import base64
from weakref import WeakValueDictionary
from IPython.display import Javascript, display
from ipywidgets import widgets
from typing import Any, Sequence, List, Dict, Tuple
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MissingnessPlotBase:
    def __init__(
        self, data: Missingness, initial_selection: Selection = Selection()
    ):
        raise NotImplementedError()

    def plot(self) -> bokeh.plotting.Figure:
        raise NotImplementedError()

    def selection_to_plot_indices(self, selection: Selection) -> Sequence[int]:
        raise NotImplementedError()

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        raise NotImplementedError()


class ValueBarChart(MissingnessPlotBase):
    def __init__(self, data: Missingness, initial_selection=Selection()):
        self._data = data

        self.source = ColumnDataSource(
            missingness.value_bar_chart_data(data).reset_index()
        )

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 0.5
        self.tools = ["box_select", "tap", "reset"]
        self.height = 960
        self.width = 960
        self.title = "Value bar chart"
        self.ylabel = "Number of missing values"

    def plot(self) -> bokeh.plotting.Figure:
        p = figure(
            title=self.title,
            tools=self.tools,
            width=self.width,
            height=self.height,
            x_range=self._data.column_labels(),
        )
        p.vbar(
            x="index", top="_count", width=self.bar_width, source=self.source
        )
        p.xaxis.major_label_orientation = "vertical"
        p.yaxis.axis_label = self.ylabel

        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        col_labels = self._data.column_labels()

        return Selection(
            columns=self._data.invert_column_selection(
                [col_labels[i] for i in indices]
            )
        )

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        col_labels = self._data.column_labels()
        col_indices_np_tuple = np.where(
            np.in1d(col_labels, selection.columns, invert=True)
        )
        col_indices = list(col_indices_np_tuple[0])

        return col_indices


class CombinationHeatmap(MissingnessPlotBase):
    def __init__(self, data: Missingness, initial_selection=Selection()):
        self._data = data

        heatmap_data = missingness.heatmap_data(data)
        heatmap_data["pattern_key_str"] = heatmap_data.index.values.astype(str)
        heatmap_data_long = pd.melt(
            heatmap_data.reset_index(),
            id_vars=["pattern_key", "pattern_key_str"],
        )
        self.source = ColumnDataSource(heatmap_data_long)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.x_range = list(heatmap_data)
        self.y_range = list(heatmap_data["pattern_key_str"].values)
        self.c_max = heatmap_data_long["value"].max()

        self.xlabel = "Fields"
        self.ylabel = "Combinations"

        self.width = 960
        self.height = 960
        self.tools = ["box_select", "tap", "reset"]
        self.fill = "#cccccc"
        self.grid_visible = False

    def plot(self) -> bokeh.plotting.Figure:
        p = figure(
            title="Combination heatmap",
            width=self.width,
            height=self.height,
            tools=self.tools,
            x_range=self.x_range,
            y_range=self.y_range,
        )
        p.background_fill_color = self.fill
        p.grid.visible = self.grid_visible

        p.rect(
            y="pattern_key_str",
            x="variable",
            source=self.source,
            width=1.0,
            height=1.0,
            fill_color=linear_cmap(
                field_name="value",
                palette=list(reversed(Oranges256)),
                low=1,
                high=self.c_max,
                # transparent
                low_color="#00000000",
            ),
            line_color=self.fill,
        )
        p.xaxis.major_label_orientation = "vertical"
        p.yaxis.major_label_text_color = None
        p.yaxis.axis_label = self.ylabel
        p.xaxis.axis_label = self.xlabel

        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        n_combinations = len(self._data.patterns())
        pattern_key = self._data.patterns().index.values
        include = list({pattern_key[i % n_combinations] for i in indices})
        exclude = self._data.invert_pattern_selection(include)

        return Selection(patterns=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        n_columns = len(self._data.column_labels())
        n_patterns = len(self._data.patterns())

        pattern_indices_tuple = np.where(
            np.in1d(
                self._data.patterns().index, selection.patterns, invert=True
            )
        )
        pattern_indices = pattern_indices_tuple[0]

        box_indices = [
            j * n_patterns + i
            for j in range(n_columns)
            for i in pattern_indices
        ]

        return box_indices


class PlotSession:
    """Class representing an interactive plotting session in a Jupyter notebook

    A session contains a sequence of named plots.  New selections can
    be made interactively from the plots, and other plots added to the
    session.

    """

    # The object holding the named plots that the user creates with
    # add_plot
    _selection_history: SelectionHistory

    # A dictionary from (name, tab_name) to a Bokeh plot object. Here
    # 'name' is the user provided name of the selection being plotted
    # (it is also a key of _selection_history) -- this identifies a
    # particular group of Tabs -- and 'tab_name' is the name of a
    # particular tab (one of "value_bar_chart", "combination_heatmap",
    # ...)
    _plots: Dict[Tuple[Any, Any], MissingnessPlotBase]

    # A class-member dictionary holding PlotSession objects, keyed
    # by their id().  This allows the JavaScript callback (that in
    # turn runs code in the IPython kernel) to look up the caller. The
    # Bokeh plot selection can then be passed to a method of the
    # corresponding object.  Uses WeakValueDictionary so that the
    # reference here does not prevent the object being garbage
    # collected.
    _instances = WeakValueDictionary()

    def __init__(self, df):
        bokeh.io.output_notebook(hide_banner=True)

        m = Missingness.from_data_frame(df)

        self._selection_history = SelectionHistory(m)
        self._plots = {}

        PlotSession._instances[id(self)] = self

    def _selection_callback(self, source, name, tabname):
        """Return a javascript callback that updates the particular
        PlotSession instance according to the selection made in a
        Bokeh plot.

        """
        return CustomJS(
            args={
                "source": source,
                "instance_id": id(self),
                "name": repr(name),
                "tabname": repr(tabname),
            },
            code="""
            var indices = source.selected.indices;
            var kernel = IPython.notebook.kernel;
            // ensure that pace.plots is imported under its canonical name
            kernel.execute("import pace.plots");
            kernel.execute(`(
                pace.plots.PlotSession
                ._instances[${instance_id}]
                ._update_selection(
                    ${name},
                    ${tabname},
                    [${indices}],
                )
            )`);
            """,
        )

    def _update_selection(self, name, tabname, indices):
        self._selection_history.update_active(
            name,
            self._plots[name, tabname].plot_indices_to_selection(indices),
        )

    def _add_subplot(self, plotter_cls, name, tabname):

        m = self._selection_history.missingness(name)
        plotter = plotter_cls(m)

        self._plots[name, tabname] = plotter

        callback = self._selection_callback(plotter.source, name, tabname)

        p = plotter.plot()
        p.js_on_event("selectiongeometry", callback)

        return p

    def add_plot(self, name, based_on=None):
        """Creates all the plot options and shows them in a tab layout"""

        self._selection_history.new_selection(name, based_on)

        p1 = self._add_subplot(ValueBarChart, name, "value_bar_chart")
        tab1 = Panel(child=p1, title="Value bar chart")

        p2 = self._add_subplot(CombinationHeatmap, name, "combination_heatmap")
        tab2 = Panel(child=p2, title="Combination heatmap")

        tabs = Tabs(tabs=[tab1, tab2], active=0)

        show(tabs)

        logging.info(
            f"""

    # ********
    # To add a plot, insert a new cell below, type "add_plot(selected_indices)" and run cell.
    # ********"""
        )
