import json
from bokeh.models.annotations import ColorBar
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
from pace.missingness import Missingness
import pace.missingness as missingness
from pace.history import SelectionHistory, Selection


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


class SetBarChart(MissingnessPlotBase):
    def __init__(self, data: Missingness, initial_selection=Selection()):
        self._data = data
        self._bar_data = missingness.set_bar_chart_data(data)
        self.source = ColumnDataSource(
            missingness.set_bar_chart_data(data).reset_index()
        )

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 0.5
        self.tools = ["box_select", "tap", "reset"]
        self.height = 960
        self.width = 960
        self.title = "Set bar chart"
        self.xlabel = "Set"
        self.ylabel = "Cardinality"

    def plot(self) -> bokeh.plotting.Figure:
        p = figure(
            title=self.title,
            tools=self.tools,
            width=self.width,
            height=self.height,
            x_range=list(self._bar_data.index,),
        )
        p.vbar(
            x="index", top="_count", width=self.bar_width, source=self.source
        )
        p.xaxis.major_label_orientation = "vertical"
        p.xaxis.axis_label = self.xlabel
        p.yaxis.axis_label = self.ylabel

        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        col_labels = self._data.columns()

        return Selection(
            columns=self._data.invert_column_selection(
                [col_labels[i] for i in indices]
            )
        )

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        col_labels = self._data.columns()
        col_indices_np_tuple = np.where(
            np.in1d(col_labels, selection.columns, invert=True)
        )
        col_indices = list(col_indices_np_tuple[0])

        return col_indices


class SetCardinalityHistogram(MissingnessPlotBase):
    def __init__(self, data: Missingness, initial_selection=Selection()):
        self._data = data
        self._bins = 11
        (
            self._hist_data,
            self._column_data_source,
            self._hist_edges,
        ) = missingness.set_cardinality_histogram_data(data, self._bins)
        self.source = ColumnDataSource(data=self._column_data_source)
        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )
        self.bar_width = 1.0
        self.tools = ["box_select", "tap", "reset"]
        self.height = 960
        self.width = 960
        self.title = "Set cardinality histogram"
        self.xlabel = "Cardinality of set"
        self.ylabel = "Number of sets with cardinality"
        self.linecolor = "white"

    def plot(self) -> bokeh.plotting.Figure:
        p = figure(
            title=self.title,
            tools=self.tools,
            width=self.width,
            height=self.height,
            # x_range=self._data.columns(),
        )
        p.xaxis.ticker = [x + 1 for x in range(self._bins)]
        p.yaxis.ticker = [
            x + 1 for x in range(self._column_data_source["_bin_count"].max())
        ]
        p.vbar(
            x="_bin_id",
            top="_bin_count",
            width=self.bar_width,
            source=self.source,
            line_color=self.linecolor,
        )
        p.xaxis.major_label_overrides = self._get_xtick_labels()
        p.xaxis.axis_label = self.xlabel
        p.yaxis.axis_label = self.ylabel

        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        bin_ids = [
            x + 1 for x in indices
        ]  # bin_ids for 10 bins are 1-10, indices plot 0-9
        include = list(
            self._hist_data[self._hist_data["_bin_id"].isin(bin_ids)].index
        )
        exclude = self._data.invert_column_selection(include)
        return Selection(columns=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        col_labels = self._data.columns()
        col_indices_np_tuple = np.where(
            np.in1d(col_labels, selection.columns, invert=True)
        )
        col_indices = list(col_indices_np_tuple[0])
        col_selection = [col_labels[i] for i in col_indices]
        indices = list(
            self._hist_data["_bin_id"][
                self._hist_data.index.isin(col_selection)
            ]
            .sort_values()
            .unique()
        )
        indices = [x - 1 for x in indices]
        return indices

    def _get_xtick_labels(self):
        keys = [str(x + 1) for x in range(self._bins)]
        vals = [
            f"{int(np.ceil(self._hist_edges[i]))} - {int(np.floor(self._hist_edges[i+1]))}"
            for i in range(len(self._hist_edges) - 1)
        ]
        vals = ["0"] + vals
        return dict(zip(keys, vals))


class IntersectionBarChart(MissingnessPlotBase):
    def __init__(self, data: Missingness, initial_selection=Selection()):
        self._data = data
        self._bar_data = missingness.combination_bar_chart_data(data)
        self.source = ColumnDataSource(self._bar_data)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 1.0
        self.tools = ["box_select", "tap", "reset"]
        self.height = 960
        self.width = 960
        self.linecolor = "white"
        self.title = "Intersection bar chart"
        self.xlabel = "Intersections"
        self.ylabel = "Number of intersections"

    def plot(self) -> bokeh.plotting.Figure:
        p = figure(
            title=self.title,
            tools=self.tools,
            width=self.width,
            height=self.height,
            # x_range=self._data.columns(),
        )
        p.vbar(
            x="index",
            top="_count",
            width=self.bar_width,
            source=self.source,
            line_color=self.linecolor,
        )
        p.xaxis.axis_label = self.xlabel
        p.yaxis.axis_label = self.ylabel
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.xaxis.major_label_text_color = None
        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        include = list(self._bar_data["combination_id"].iloc[indices])
        exclude = self._data.invert_combination_selection(include)
        return Selection(combinations=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        combination_ids_tuple = np.where(
            np.in1d(
                self._data.combinations().index,
                selection.combinations,
                invert=True,
            )
        )
        combination_ids = combination_ids_tuple[0]

        bar_indices = list(
            self._bar_data[
                self._bar_data["combination_id"].isin(combination_ids)
            ].index
        )
        return bar_indices


class IntersectionCardinalityHistogram(MissingnessPlotBase):
    def __init__(self, data: Missingness, initial_selection=Selection()):
        self._data = data
        self._bins = 10
        (
            self._hist_data,
            self._column_data_source,
            self._hist_edges,
        ) = missingness.combination_count_histogram_data(data, self._bins)
        self.source = ColumnDataSource(self._column_data_source)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 1.0
        self.tools = ["box_select", "tap", "reset"]
        self.height = 960
        self.width = 960
        self.linecolor = "white"
        self.title = "Intersection cardinality histogram"
        self.xlabel = "Cardinality"
        self.ylabel = "Number of intersections"

    def plot(self) -> bokeh.plotting.Figure:
        p = figure(
            title=self.title,
            tools=self.tools,
            width=self.width,
            height=self.height,
            # x_range=self._data.columns(),
        )
        p.vbar(
            x="_bin",
            top="_count",
            width=self.bar_width,
            source=self.source,
            line_color=self.linecolor,
        )
        p.xaxis.ticker = [x for x in range(self._bins)]
        p.xaxis.major_label_overrides = self._get_xtick_labels()
        p.xaxis.axis_label = self.xlabel
        p.yaxis.axis_label = self.ylabel
        # fix xaxis ticklabels
        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        include = list(
            self._hist_data[self._hist_data["_bin_id"].isin(indices)].index
        )
        exclude = self._data.invert_combination_selection(include)
        return Selection(combinations=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        combination_ids_tuple = np.where(
            np.in1d(
                self._data.combinations().index,
                selection.combinations,
                invert=True,
            )
        )
        combination_ids = combination_ids_tuple[0]
        indices = list(self._hist_data["_bin_id"].iloc[combination_ids])
        return indices

    def _get_xtick_labels(self):
        key_list = [str(x) for x in range(self._bins)]
        xtick_labels = dict.fromkeys(key_list)
        for i in range(self._bins):  # write as list comprehension?
            left = int(np.ceil(self._hist_edges[i]))
            right = int(np.floor(self._hist_edges[i + 1]))
            xtick_labels[str(i)] = f"{left}-{right}"
        return xtick_labels


class IntersectionDegreeHistogram(MissingnessPlotBase):
    def __init__(self, data: Missingness, initial_selection=Selection()):
        self._data = data
        self._bins = 10
        (
            self._hist_data,
            self._column_data_source,
            self._hist_edges,
        ) = missingness.combination_length_histogram_data(data, self._bins)
        self.source = ColumnDataSource(self._column_data_source)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 1.0
        self.tools = ["box_select", "tap", "reset"]
        self.height = 960
        self.width = 960
        self.linecolor = "white"
        self.title = "Intersection degree histogram"
        self.xlabel = "Intersection Degree"
        self.ylabel = "Number of intersections"

    def plot(self) -> bokeh.plotting.Figure:
        p = figure(
            title=self.title,
            tools=self.tools,
            width=self.width,
            height=self.height,
            # x_range=self._data.columns(),
        )
        p.vbar(
            x="_bin",
            top="_count",
            width=self.bar_width,
            source=self.source,
            line_color=self.linecolor,
        )
        p.xaxis.ticker = [x for x in range(self._bins)]
        p.xaxis.major_label_overrides = self._get_xtick_labels()
        p.xaxis.axis_label = self.xlabel
        p.yaxis.axis_label = self.ylabel
        # TODO: fix xaxis ticklabels, fix ticks yaxis
        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        include = list(
            self._hist_data[self._hist_data["_bin_id"].isin(indices)].index
        )
        exclude = self._data.invert_combination_selection(include)
        return Selection(combinations=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        combination_ids_tuple = np.where(
            np.in1d(
                self._data.combinations().index,
                selection.combinations,
                invert=True,
            )
        )
        combination_ids = combination_ids_tuple[0]
        indices = list(self._hist_data["_bin_id"].iloc[combination_ids])
        return indices

    def _get_xtick_labels(self):
        key_list = [str(x) for x in range(self._bins)]
        xtick_labels = dict.fromkeys(key_list)
        for i in range(self._bins):  # write as list comprehension
            left = int(np.ceil(self._hist_edges[i]))
            right = int(np.floor(self._hist_edges[i + 1]))
            xtick_labels[str(i)] = f"{left}-{right}"
        return xtick_labels


class IntersectionHeatmap(MissingnessPlotBase):
    def __init__(self, data: Missingness, initial_selection=Selection()):
        self._data = data

        heatmap_data = missingness.intersection_heatmap_data(data)
        heatmap_data["combination_id_str"] = heatmap_data.index.values.astype(
            str
        )
        heatmap_data_long = pd.melt(
            heatmap_data.reset_index(),
            id_vars=["combination_id", "combination_id_str"],
        )
        self.source = ColumnDataSource(heatmap_data_long)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.x_range = list(heatmap_data)
        self.y_range = list(heatmap_data["combination_id_str"].values)
        self.c_max = heatmap_data_long["value"].max()

        self.xlabel = "Sets"
        self.ylabel = "Intersections"

        self.width = 960
        self.height = 960
        self.tools = ["box_select", "tap", "reset"]
        self.fill = "#cccccc"
        self.grid_visible = False

    def plot(self) -> bokeh.plotting.Figure:
        p = figure(
            title="Intersection heatmap",
            width=self.width,
            height=self.height,
            tools=self.tools,
            x_range=self.x_range,
            y_range=self.y_range,
        )
        p.background_fill_color = self.fill
        p.grid.visible = self.grid_visible
        mapper = linear_cmap(
            field_name="value",
            palette=list(reversed(Oranges256)),
            low=1,
            high=self.c_max,
            # transparent
            low_color="#00000000",
        )
        p.rect(
            y="combination_id_str",
            x="variable",
            source=self.source,
            width=1.0,
            height=1.0,
            fill_color=mapper,
            line_color=self.fill,
        )
        p.xaxis.major_label_orientation = "vertical"
        p.yaxis.major_label_text_color = None
        p.yaxis.axis_label = self.ylabel
        p.xaxis.axis_label = self.xlabel
        color_bar = ColorBar(
            color_mapper=mapper["transform"],
            width=10,
            title="Number of records",
        )
        p.add_layout(color_bar, "right")
        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        n_combinations = len(self._data.combinations())
        combination_id = self._data.combinations().index.values
        include = list({combination_id[i % n_combinations] for i in indices})
        exclude = self._data.invert_combination_selection(include)

        return Selection(combinations=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        n_columns = len(self._data.columns())
        n_combinations = len(self._data.combinations())

        combination_ids_tuple = np.where(
            np.in1d(
                self._data.combinations().index,
                selection.combinations,
                invert=True,
            )
        )
        combination_ids = combination_ids_tuple[0]

        box_indices = [
            j * n_combinations + i
            for j in range(n_columns)
            for i in combination_ids
        ]

        return box_indices


class PlotSession:
    """An interactive plotting session in a Jupyter notebook

    A session contains a sequence of named selections, each with a few
    corresponding Bokeh plots (tabs in a tabbed layout).  New
    selections can be made interactively from the plots, and then
    other plots added.

    """

    # The object holding the named plots that the user creates with
    # add_plot
    _selection_history: SelectionHistory

    # A dictionary from (name, tab_name) to a MissingnessPlotBase
    # object. Here 'name' is the user provided name of the selection
    # being plotted (it is also a key of _selection_history) -- this
    # identifies a particular group of Tabs -- and 'tab_name' is the
    # name of a particular tab (one of "value_bar_chart",
    # "combination_heatmap", ...)
    #
    # The Bokeh plot object (produced by calling .plot()) is not
    # stored -- this would not be particularly useful anyway, since it
    # is not the object visible in the notebook!
    _plots: Dict[Tuple[Any, Any], MissingnessPlotBase]

    # For each plot, an integer representing the tab that should be
    # visible (used for saving/loading sessions)
    _active_tabs: Dict[Any, int]

    # A class-member dictionary holding PlotSession objects, keyed
    # by their id().  This allows the JavaScript callback (that in
    # turn runs code in the IPython kernel) to look up the caller. The
    # Bokeh plot selection can then be passed to a method of the
    # corresponding object.  Uses WeakValueDictionary so that the
    # reference here does not prevent the object being garbage
    # collected.
    _instances = WeakValueDictionary()

    def __init__(self, df, session_file=None, set_mode=False, verbose=False):
        self._verbose = verbose

        bokeh.io.output_notebook(hide_banner=True)

        m = Missingness.from_data_frame(df, set_mode=set_mode)

        self._plots = {}

        if session_file is None:
            self._selection_history = SelectionHistory(m)
            self._active_tabs = {}

        else:
            with open(session_file) as f:
                j = json.load(f)

            self._selection_history = SelectionHistory(
                m, j["selection_history"]
            )
            self._active_tabs = j["active_tabs"]

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
            console.log(indices)
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

    def _active_tab_callback(self, name):
        """Return a javascript callback that updates the dictionary indicating
        the active tab in each Bokeh Tabs panel"""
        return CustomJS(
            args={"instance_id": id(self), "name": repr(name)},
            code="""
            var kernel = IPython.notebook.kernel;
            var active = cb_obj.active;
            kernel.execute("import pace.plots");
            kernel.execute(`pace.plots.PlotSession._instances[
                                ${instance_id}
                            ]._active_tabs[${name}] = ${active}
            `);
            """,
        )

    def _update_selection(self, name, tabname, indices):
        self._selection_history[name] = self._plots[
            name, tabname
        ].plot_indices_to_selection(indices)

    def _add_subplot(self, plotter_cls, name, tabname):
        parent = self._selection_history.parent(name)

        m = self._selection_history.missingness(parent)
        selection = self._selection_history[name]

        plotter = plotter_cls(m, initial_selection=selection)

        self._plots[name, tabname] = plotter

        callback = self._selection_callback(plotter.source, name, tabname)

        p = plotter.plot()
        p.js_on_event("selectiongeometry", callback)

        return p

    def add_plot(self, name, based_on=None):
        """Creates all the plot options and shows them in a tab layout"""

        self._selection_history.new_selection(name, based_on)

        if name not in self._active_tabs:
            self._active_tabs[name] = 0

        p1 = self._add_subplot(SetBarChart, name, "set_bar_chart")
        tab1 = Panel(child=p1, title="Set bar chart")

        p2 = self._add_subplot(
            SetCardinalityHistogram, name, "set_cardinality_histogram"
        )
        tab2 = Panel(child=p2, title="Set cardinality histogram")

        p3 = self._add_subplot(
            IntersectionHeatmap, name, "intersection_heatmap"
        )
        tab3 = Panel(child=p3, title="Intersection heatmap")

        p4 = self._add_subplot(
            IntersectionBarChart, name, "intersection_bar_chart"
        )
        tab4 = Panel(child=p4, title="Intersection bar chart")

        p5 = self._add_subplot(
            IntersectionCardinalityHistogram,
            name,
            "intersection_cardinality_histogram",
        )
        tab5 = Panel(child=p5, title="Intersection cardinality histogram")

        p6 = self._add_subplot(
            IntersectionDegreeHistogram, name, "intersection_degree_histogram"
        )
        tab6 = Panel(child=p6, title="Intersection degree histogram")

        tabs = Tabs(
            tabs=[tab1, tab2, tab3, tab4, tab5, tab6],
            active=self._active_tabs[name],
        )
        tabs.js_on_change("active", self._active_tab_callback(name))

        show(tabs)

        if self._verbose:
            logging.info(
                f"""

    # ********
    # To add a plot, insert a new cell below, type "add_plot(selected_indices)" and run cell.
    # ********"""
            )

    def add_selection(
        self,
        name,
        based_on=None,
        columns=None,
        combinations=None,
        records=None,
        invert=False,
    ):
        if not invert:
            parent = self._selection_history.missingness(based_on)
            columns = parent.invert_column_selection(columns)
            combinations = parent.invert_combination_selection(combinations)
            records = parent.invert_record_selection(records)

        selection = Selection(
            columns=columns, combinations=combinations, records=records
        )

        self._selection_history.new_selection(name, based_on)
        self._selection_history[name] = selection

    def selected_records(self, name=None, base_selection=None):
        return self._selection_history.selected_records(name, base_selection)

    def missingness(self, name=None):
        return self._selection_history.missingness(name)

    def dict(self):
        """Returns a json-serializable dict representing the session

        It includes:
          - the plot selections (contained in _selection_history)
          - the active (currently-selected) tab in each Bokeh 'Tabs' layout

        It does not include any of the missingness data itself

        This is used by .save() to save the session state to a file.
        """

        return {
            "selection_history": self._selection_history.dict(),
            "active_tabs": self._active_tabs,
        }

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.dict(), f, cls=NPIntEncoder, indent=1)


class NPIntEncoder(json.JSONEncoder):
    """JSON encoder for numpy integer dtypes"""

    def default(self, obj):
        if np.issubdtype(obj, np.integer):
            return int(obj)
        return super().default(obj)