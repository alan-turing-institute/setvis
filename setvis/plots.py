import json
from bokeh.models.annotations import ColorBar
import bokeh.plotting
from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource,
    DataRange1d,
    HelpTool,
)
from bokeh.palettes import Oranges256
from bokeh.transform import linear_cmap
from bokeh.models.widgets import Panel, Tabs
from bokeh.events import SelectionGeometry
import bokeh.io
import bokeh.server
import pandas as pd
import numpy as np
import logging

from typing import Any, Sequence, List, Dict, Tuple
from .membership import Membership

# import .membership as membership
from .history import SelectionHistory, Selection
import setvis.plotutils as plotutils


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PlotBase:
    def __init__(
        self, data: Membership, initial_selection: Selection = Selection()
    ):
        raise NotImplementedError()

    def plot(self) -> bokeh.plotting.Figure:
        raise NotImplementedError()

    def selection_to_plot_indices(self, selection: Selection) -> Sequence[int]:
        """Maps a ``Selection`` to the corresponding bin indices of the
        histogram.

        Setvis understands items selected in a ``Membership`` object in form
        of records, columns or intersections, while Bokeh uses numeric indices to
        identify the elements of a plot (e.g. bars, fields in a heatmap).
        This converts such setvis ``Selection`` object into a list of corresponding
        Bokeh indices. This conversion is specific to a plot type.

        Parameters
        ----------

        selection : Selection
            selected items of ``Membership`` object

        Returns
        -------

        Sequence[int]
            Bokeh indices of the plot elements that correspond to the setvis selection.

        Raises
        ------

        NotImplementedError
            _description_
        """
        raise NotImplementedError()

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        """Maps the interactive selection made in the plot to
        a ``Selection`` object.

        To identify the elements of a plot (e.g. bars, fields in a heatmap),
        Bokeh indexes them in a numeric way. This converts the
        list of selected Bokeh indices into a selection of records, columns
        or intersections understood by setvis. This conversion is specific to
        a plot type.

        Parameters
        ----------
        indices : Sequence[int]
            Bokeh indices of the elements selected in the plot

        Returns
        -------
        Selection
            items in Membership object that correspond to the plot selection
        """
        raise NotImplementedError()


class SetBarChart(PlotBase):
    """A bar chart plot representing the number of records in each set.

    Only the items in the initial selection of the Membership object are
    included in the plot. If no selection is specified, all items are included.

    Parameters
    ----------
    data : Membership
        a :class:`~setvis.membership.Membership` object
    initial_selection : Selection
        initial selection of items in the membership object to be included in
        the set bar chart
    sort_x_by: str
        Name of the sort option for the x-axis.
        Sort options are:
        - "value": sorts the bars along the x-axis with ascending or descending
        y-value as specified in `sort_x_order`
        - "alphabetical": sorts the bars along the x-axis in alphabetical order
        as specified in `sort_x_order`
        - default: if none of the above is provided the bars are sorted
        in the order they appear in the dataset.
    sort_x_order: str
        Sorting order for the x-axis. Options are:
        - "ascending"
        - "descending"
    """

    def __init__(
        self,
        data: Membership,
        initial_selection=Selection(),
        sort_x_by=None,
        sort_y_by=None,
        sort_x_order=None,
        sort_y_order=None,
        bins=None,
    ):
        set_mode = data._set_mode
        self._data = data
        self._bar_data = plotutils.set_bar_chart_data(
            data, sort_x_by, sort_x_order
        )
        self.source = ColumnDataSource(
            plotutils.set_bar_chart_data(data).reset_index()
        )

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 0.5
        self.tools = [
            "box_select",
            "tap",
            "box_zoom",
            "pan",
            "reset",
            "save",
            HelpTool(
                redirect="https://setvis.readthedocs.io/en/latest/plots.html#setvis.plots.IntersectionBarChart",
                description="SetBarChart",
            ),
        ]
        self.title = "Set bar chart" if set_mode else "Value bar chart"
        self.xlabel = "Set" if set_mode else "Fields"
        self.ylabel = "Cardinality" if set_mode else "Number of missing values"

    def plot(self, **kwargs) -> bokeh.plotting.Figure:
        """Creates a figure with the set bar chart plot.

        Parameters
        ----------
        title : str
            title of the plot.
        tools : list[str]
            list of tools to interact with Bokeh plot. For each tool an
            icon appears in the plot toolbar.
        x_range : range of x-axis. By default, range is automatically
            determined based on plot data.
        y_range : range of y-axis. By default, range is automatically
            determined based on plot data.
        y_axis_type : str
            Options are "linear" and "log". The default is "linear".
        **kwargs
          All other arguments are forwarded to :func:`bokeh.plotting.figure`

        Returns
        -------
        bokeh.plotting.Figure
            bar chart plot
        """
        kwargs.setdefault("title", self.title)
        kwargs.setdefault("tools", self.tools)
        kwargs.setdefault("x_range", list(self._bar_data.index))

        if kwargs.get("y_axis_type", "linear") == "log":
            kwargs.setdefault("y_range", DataRange1d(start=0.5))

        p = figure(**kwargs)
        p.vbar(
            x="index",
            bottom=0.001,
            top="_count",
            width=self.bar_width,
            source=self.source,
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


class SetCardinalityHistogram(PlotBase):
    """A histogram plot that bins the number of records in a set.

    The first bin only contains the sets with no records in them.

    Only the items in the initial selection of the Membership object are
    included in the plot. If no selection is specified, all items are included.

    Parameters
    ----------
    data : Membership
        a :class:`~setvis.membership.Membership` object
    initial_selection : Selection
        initial selection of items in the membership object to be included
        in the histogram
    bins : int
        number of histogram bins
    """

    def __init__(
        self,
        data: Membership,
        initial_selection=Selection(),
        sort_x_by=None,
        sort_y_by=None,
        sort_x_order=None,
        sort_y_order=None,
        bins=None,
    ):
        set_mode = data._set_mode
        self._data = data
        if not bins:
            bins = 11
        self._bins = bins
        (
            self._hist_data,
            self._column_data_source,
            self._hist_edges,
        ) = plotutils.set_cardinality_histogram_data(
            data,
            bins=self._bins,
        )
        self.source = ColumnDataSource(data=self._column_data_source)
        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )
        self.bar_width = 1.0
        self.tools = ["box_select", "tap", "reset", "save"]
        self.title = (
            "Set cardinality histogram"
            if set_mode
            else "Value count histogram"
        )
        self.xlabel = (
            "Cardinality of set" if set_mode else "Number of missing values"
        )
        self.ylabel = (
            "Number of sets with cardinality"
            if set_mode
            else "Number of fields"
        )
        self.linecolor = "white"

    def plot(self, **kwargs) -> bokeh.plotting.Figure:
        """Creates a figure with the set cardinality histogram plot.

        Parameters
        ----------
        title : str
            title of the plot.
        tools : list[str]
            list of tools to interact with Bokeh plot. For each tool an
            icon appears in the plot toolbar.
        y_range : range of y-axis. By default, range is automatically
            determined based on plot data.
        y_axis_type : str
            Options are "linear" and "log". The default is "linear".
        **kwargs
            All other arguments are forwarded to :func:`bokeh.plotting.figure`

        Returns
        -------
        bokeh.plotting.Figure
            histogram plot
        """
        kwargs.setdefault("title", self.title)
        kwargs.setdefault("tools", self.tools)

        if kwargs.get("y_axis_type", "linear") == "log":
            kwargs.setdefault("y_range", DataRange1d(start=0.5))

        p = figure(**kwargs)

        p.xaxis.ticker = [x + 1 for x in range(self._bins)]
        p.yaxis.ticker = [
            x + 1 for x in range(self._column_data_source["_bin_count"].max())
        ]
        p.vbar(
            x="_bin_id",
            bottom=0.001,
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
        """Returns the labels for the x-axis ticks of the histogram.

        Note that the first bin is a special case and only contains sets
        without members.

        Returns
        -------
        dict
            dict that maps the bin index to the tick labels
        """
        keys = [str(x + 1) for x in range(self._bins)]
        vals = [
            f"{int(np.ceil(self._hist_edges[i]))} - {int(np.floor(self._hist_edges[i+1]))}"
            for i in range(len(self._hist_edges) - 1)
        ]
        vals = ["0"] + vals
        return dict(zip(keys, vals))


class IntersectionBarChart(PlotBase):
    """A bar chart plot that represents the number of records for each unique
    set intersection.

    Only the items in the initial selection of the Membership object are
    included in the plott. If no selection is specified,
    all items are included.

    Parameters
    ----------
    data : Membership
        a :class:`~setvis.membership.Membership` object
    initial_selection : Selection
        initial selection of items in the membership object to be included in
        the set bar chart
    sort_x_by: str
        Name of the sort option for the x-axis.
        Sort options are:
        - "value": sorts the bars along the x-axis with ascending or descending
        y-value as specified in `sort_x_order`
        - "alphabetical": sorts the bars along the x-axis in alphabetical order
        as specified in `sort_x_order`
        - default: if none of the above is provided the bars are sorted
        in the order they appear in the dataset.
    sort_x_order: str
        Sorting order for the x-axis. Options are:
        - "ascending"
        - "descending"
    """

    def __init__(
        self,
        data: Membership,
        initial_selection=Selection(),
        sort_x_by=None,
        sort_y_by=None,
        sort_x_order=None,
        sort_y_order=None,
        bins=None,
    ):
        set_mode = data._set_mode
        self._data = data
        self._bar_data = plotutils.intersection_bar_chart_data(
            data,
            sort_x_by,
            sort_x_order,
        )
        self.source = ColumnDataSource(self._bar_data)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 1.0
        self.tools = ["box_select", "tap", "reset", "save"]
        self.linecolor = "white"
        self.title = (
            "Intersection bar chart" if set_mode else "Combination bar chart"
        )
        self.xlabel = "Intersections" if set_mode else "Combinations"
        self.ylabel = (
            "Number of intersections" if set_mode else "Number of records"
        )

    def plot(self, **kwargs) -> bokeh.plotting.Figure:
        """Creates a figure with the intersection bar chart plot.

        Parameters
        ----------
        title : str
            title of the plot
        tools : list[str]
            list of tools to interact with Bokeh plot. For each tool an
            icon appears in the plot toolbar.
        y_range : range of y-axis. By default, range is automatically
            determined based on plot data.
        y_axis_type : str
            Options are "linear" and "log". The default is "linear".
        **kwargs
            All other arguments are forwarded to :func:`bokeh.plotting.figure`

        Returns
        -------
        bokeh.plotting.Figure
            bar chart plot. The default is
        """
        kwargs.setdefault("title", self.title)
        kwargs.setdefault("tools", self.tools)

        if kwargs.get("y_axis_type", "linear") == "log":
            kwargs.setdefault("y_range", DataRange1d(start=0.5))

        p = figure(**kwargs)

        p.vbar(
            x="index",
            bottom=0.001,
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
        include = list(self._bar_data["intersection_id"].iloc[indices])
        exclude = self._data.invert_intersection_selection(include)
        return Selection(intersections=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        intersection_ids_tuple = np.where(
            np.in1d(
                self._data.intersections().index,
                selection.intersections,
                invert=True,
            )
        )
        intersection_ids = intersection_ids_tuple[0]

        bar_indices = list(
            self._bar_data[
                self._bar_data["intersection_id"].isin(intersection_ids)
            ].index
        )
        return bar_indices


class IntersectionCardinalityHistogram(PlotBase):
    """A histogram plot that bins the number of records for each
    set intersection.

    Only the items in the initial selection of the Membership object are
    included in the plot. If no selection is specified, all items are included.

    Parameters
    ----------
    data : Membership
        a :class:`~setvis.membership.Membership` object
    initial_selection : Selection
        initial selection of items in the membership object to be included in
        the intersection cardinality histogram

    bins : int

    """

    def __init__(
        self,
        data: Membership,
        initial_selection=Selection(),
        sort_x_by=None,
        sort_y_by=None,
        sort_x_order=None,
        sort_y_order=None,
        bins=None,
    ):
        if not bins:
            bins = 10

        set_mode = data._set_mode
        self._data = data
        (
            self._hist_data,
            self._column_data_source,
            self._hist_edges,
            self._bins,
        ) = plotutils.intersection_cardinality_histogram_data(data, bins=bins)
        self.source = ColumnDataSource(self._column_data_source)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 1.0
        self.tools = ["box_select", "tap", "reset", "save"]
        self.height = 960
        self.width = 960
        self.linecolor = "white"
        self.title = (
            "Intersection cardinality histogram"
            if set_mode
            else "Combination count histogram"
        )
        self.xlabel = "Cardinality" if set_mode else "Number of records"
        self.ylabel = (
            "Number of intersections" if set_mode else "Number of combinations"
        )

    def plot(self, **kwargs) -> bokeh.plotting.Figure:
        """Creates a figure with the intersection cardinality histogram plot.

        Parameters
        ----------
        title : str
            title of the plot
        tools : list[str]
            list of tools to interact with Bokeh plot. For each tool an
            icon appears in the plot toolbar.
        y_range : range of y-axis. By default, range is automatically
            determined based on plot data.
        y_axis_type : str
            Options are "linear" and "log". The default is "linear".
        **kwargs
            All other arguments are forwarded to :func:`bokeh.plotting.figure`

        Returns
        -------
        bokeh.plotting.Figure
            histogram plot
        """
        kwargs.setdefault("title", self.title)
        kwargs.setdefault("tools", self.tools)

        if kwargs.get("y_axis_type", "linear") == "log":
            kwargs.setdefault("y_range", DataRange1d(start=0.5))

        p = figure(**kwargs)

        p.vbar(
            x="_bin",
            bottom=0.001,
            top="_count",
            width=self.bar_width,
            source=self.source,
            line_color=self.linecolor,
        )
        p.xaxis.ticker = [x - 0.5 for x in range(self._bins)]
        p.xaxis.major_label_overrides = self._get_xtick_labels()
        p.xaxis.axis_label = self.xlabel
        p.yaxis.axis_label = self.ylabel
        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        include = list(
            self._hist_data[self._hist_data["_bin_id"].isin(indices)].index
        )
        exclude = self._data.invert_intersection_selection(include)
        return Selection(intersections=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        intersection_ids_tuple = np.where(
            np.in1d(
                self._data.intersections().index,
                selection.intersections,
                invert=True,
            )
        )
        intersection_ids = intersection_ids_tuple[0]
        indices = list(self._hist_data["_bin_id"].iloc[intersection_ids])
        return indices

    def _get_xtick_labels(self):
        keys = [str(x - 0.5) for x in range(self._bins)]
        vals = [
            f"{int(np.ceil(self._hist_edges[i]))}"
            for i in range(len(self._hist_edges))
        ]
        return dict(zip(keys, vals))


class IntersectionDegreeHistogram(PlotBase):
    """A histogram plot that bins the number of sets that form a set intersection.

    Only the items in the initial selection of the Membership object are
    included in the plot. If no selection is
    specified, all items are included.

    Parameters
    ----------
    data : Membership
        a :class:`~setvis.membership.Membership` object
    initial_selection : Selection
        initial selection of items in the membership object to be included in
        the histogram
    bins : int
        number of histogram bins
    """

    def __init__(
        self,
        data: Membership,
        initial_selection=Selection(),
        sort_x_by=None,
        sort_y_by=None,
        sort_x_order=None,
        sort_y_order=None,
        bins=None,
    ):
        set_mode = data._set_mode
        if not bins:
            bins = 10
        self._data = data
        (
            self._hist_data,
            self._column_data_source,
            self._hist_edges,
            self._bins,
        ) = plotutils.intersection_degree_histogram_data(data, bins=bins)
        self.source = ColumnDataSource(self._column_data_source)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.bar_width = 1.0
        self.tools = ["box_select", "tap", "reset", "save"]
        self.height = 960
        self.width = 960
        self.linecolor = "white"
        self.title = (
            "Intersection degree histogram"
            if set_mode
            else "Combination length histogram"
        )
        self.xlabel = (
            "Intersection Degree" if set_mode else "Combination length"
        )
        self.ylabel = (
            "Number of intersections" if set_mode else "Number of combinations"
        )

    def plot(self, **kwargs) -> bokeh.plotting.Figure:
        """Creates a figure with the intersection degree histogram.

        Parameters
        ----------
        title : str
            title of the plot
        tools : list[str]
            list of tools to interact with Bokeh plot. For each tool an
            icon appears in the plot toolbar.
        y_range : range of y-axis. By default, range is automatically
            determined based on plot data.
        y_axis_type : str
            Options are "linear" and "log". The default is "linear".
        **kwargs
            All other arguments are forwarded to :func:`bokeh.plotting.figure`

        Returns
        -------
        bokeh.plotting.Figure
            histogram plot
        """
        kwargs.setdefault("title", self.title)
        kwargs.setdefault("tools", self.tools)

        if kwargs.get("y_axis_type", "linear") == "log":
            kwargs.setdefault("y_range", DataRange1d(start=0.5))

        p = figure(**kwargs)

        p.vbar(
            x="_bin",
            bottom=0.001,
            top="_count",
            width=self.bar_width,
            source=self.source,
            line_color=self.linecolor,
        )
        p.xaxis.ticker = [x - 0.5 for x in range(self._bins)]
        p.xaxis.major_label_overrides = self._get_xtick_labels()
        p.xaxis.axis_label = self.xlabel
        p.yaxis.axis_label = self.ylabel
        return p

    def plot_indices_to_selection(self, indices: Sequence[int]) -> Selection:
        include = list(
            self._hist_data[self._hist_data["_bin_id"].isin(indices)].index
        )
        exclude = self._data.invert_intersection_selection(include)
        return Selection(intersections=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        intersection_ids_tuple = np.where(
            np.in1d(
                self._data.intersections().index,
                selection.intersections,
                invert=True,
            )
        )
        intersection_ids = intersection_ids_tuple[0]
        indices = list(self._hist_data["_bin_id"].iloc[intersection_ids])
        return indices

    def _get_xtick_labels(self):
        """Returns the labels for the x-axis ticks of the histogram.

        Returns
        -------
        dict
            a dictionary that maps the tick position on the x-axis to the
            labels
        """
        keys = [str(x - 0.5) for x in range(self._bins)]
        vals = [
            f"{int(np.ceil(self._hist_edges[i]))}"
            for i in range(len(self._hist_edges))
        ]
        return dict(zip(keys, vals))


class IntersectionHeatmap(PlotBase):
    """A heatmap that displays a matrix of sets on the x-axis and
    the set intersections on the y-axis.

    The number of records that are associated with a set intersection is
    encoded in the colour map.

    Only the items in the initial selection of the Membership object are
    included in the plot. If no selection is specified, all items are
    included.

    Parameters
    ----------
    data : Membership
        a :class:`~setvis.membership.Membership` object
    initial_selection : Selection
        initial selection of items in the membership object to be included in
        the set bar chart
    sort_x_by: str
        Name of the sort option for the x-axis.
        Sort options are:
        - "alphabetical": sorts the fields along the x-axis in alphabetical
          order as specified in `sort_x_order`
        - default: if none of the above is provided the fields on the x-axis
          of the heatmap are sorted in the order they appear in the dataset.
    sort_y_by: str
        Name of the sort option for the y-axis.
        Sort options are:
        - "value": sorts the fields along the y-axis by the heatmap value with
          the order as specified in `sort_x_order`
        - "length": sorts the fields along the y-axis by the intersection
          length with the order as specified in `sort_x_order`
        - default: if none of the above is provided the intersections on the
          y-axis of the heatmap are sorted in the order they appear in the
          dataset.
    sort_x_order: str
        - "ascending" (default)
        - "descending"
    sort_y_order: str
        - "ascending" (default)
        - "descending"
    """

    def __init__(
        self,
        data: Membership,
        initial_selection=Selection(),
        sort_x_by=None,
        sort_y_by=None,
        sort_x_order=None,
        sort_y_order=None,
        bins=None,
    ):
        set_mode = data._set_mode
        self._data = data

        heatmap_data = plotutils.intersection_heatmap_data(
            data,
            sort_x_by,
            sort_y_by,
            sort_x_order,
            sort_y_order,
        )
        heatmap_data["intersection_id_str"] = heatmap_data.index.values.astype(
            str
        )
        heatmap_data = heatmap_data.set_index("intersection_id_str")
        heatmap_data_long = pd.melt(
            heatmap_data.reset_index(),
            id_vars=["intersection_id_str"],
        )
        self.source = ColumnDataSource(heatmap_data_long)

        self.source.selected.indices = self.selection_to_plot_indices(
            initial_selection
        )

        self.x_range = list(heatmap_data)
        self.y_range = list(heatmap_data.index.values)
        self.c_max = heatmap_data_long["value"].max()
        self.title = (
            "Intersection heatmap" if set_mode else "Combination heatmap"
        )
        self.xlabel = "Sets" if set_mode else "Fields"
        self.ylabel = "Intersections" if set_mode else "Combinations"

        self.width = 960
        self.height = 960
        self.tools = ["box_select", "tap", "reset", "save"]
        self.fill = "#cccccc"
        self.grid_visible = False

    def plot(self, **kwargs) -> bokeh.plotting.Figure:
        """Creates a figure with the intersection heatmap plot.

        Parameters
        ----------
        title : str
            title of the plot
        tools : list[str]
            list of tools to interact with Bokeh plot. For each tool an
            icon appears in the plot toolbar.
        x_range : range of x-axis. By default, range is automatically
            determined based on plot data.
        y_range : range of y-axis. By default, range is automatically
            determined based on plot data.
        **kwargs
            All other arguments are forwarded to :func:`bokeh.plotting.figure`

        Returns
        -------
        bokeh.plotting.Figure
            heatmap plot
        """
        kwargs.setdefault("title", self.title)
        kwargs.setdefault("tools", self.tools)
        kwargs.setdefault("x_range", self.x_range)
        kwargs.setdefault("y_range", self.y_range)
        p = figure(**kwargs)

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
            y="intersection_id_str",
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
        n_intersections = len(self._data.intersections())
        intersection_id = self._data.intersections().index.values
        include = list({intersection_id[i % n_intersections] for i in indices})
        exclude = self._data.invert_intersection_selection(include)

        return Selection(intersections=exclude)

    def selection_to_plot_indices(self, selection: Selection) -> List[int]:
        n_columns = len(self._data.columns())
        n_intersections = len(self._data.intersections())

        intersection_ids_tuple = np.where(
            np.in1d(
                self._data.intersections().index,
                selection.intersections,
                invert=True,
            )
        )
        intersection_ids = intersection_ids_tuple[0]

        box_indices = [
            j * n_intersections + i
            for j in range(n_columns)
            for i in intersection_ids
        ]

        return box_indices


class PlotSession:
    """An interactive plotting session in a Jupyter notebook

    A session contains a sequence of named selections, each with a few
    corresponding Bokeh plots (tabs in a tabbed layout). New
    selections can be made interactively from the plots, and then
    other plots added.

    It is possible to save a session, i.e., the user-made interactive
    selections in every plot to a file, and load it to restore the state
    of the session.


    Parameters
    ----------
    data : pd.DataFrame or Membership
        contains the dataset a ``Membership`` object
    session_file : str
        file containing the interactive selections of a previously saved
        session, by default None
    set_mode : bool
        whether to operate in set mode (True) or missingness mode (False),
        by default False
    verbose : bool
        whether to produce verbose logging output, by default False

    """

    # The object holding the named plots that the user creates with
    # add_plot
    _selection_history: SelectionHistory

    # For each plot, an integer representing the tab that should be
    # visible (used for saving/loading sessions)
    _active_tabs: Dict[Any, int]

    def __init__(self, data, session_file=None, set_mode=False, verbose=False):
        self._verbose = verbose
        self._set_mode = set_mode

        bokeh.io.output_notebook(hide_banner=True)

        if isinstance(data, pd.DataFrame):
            # NB: this is for the ordinary (expanded format).
            # from_membership_data_frame() would need to be called if the data
            # frame is in the compact format
            m = Membership.from_data_frame(data, set_mode=self._set_mode)
        else:
            m = data

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

    def _add_subplot(self, plotter_cls, name, tabname, **kwargs):
        """Creates a new subplot to be added to tabbed layout

        Parameters
        ----------
        plotter_cls : class(PlotBase)
            class of the subplot object to be added.
            Options are:
            - SetBarChart
            - IntersectionHeatmap
            - SetCardinalityHistogram
            - IntersectionBarChart
            - IntersectionCardinalityHistogram
            - IntersectionDegreeHistogram
        name : str
            name of plot to which subplot is added
        tabname : str
            name of the tab

        Returns
        -------
        bokeh.plotting.Figure
            plot to be added to tabbed layout
        """
        parent = self._selection_history.parent(name)

        m = self._selection_history.membership(parent)
        selection = self._selection_history[name]

        sort_x_by = kwargs.pop("sort_x_by", None)
        sort_x_order = kwargs.pop("sort_x_order", None)
        sort_y_by = kwargs.pop("sort_y_by", None)
        sort_y_order = kwargs.pop("sort_y_order", None)
        bins = kwargs.pop("bins", None)

        plotter = plotter_cls(
            m,
            initial_selection=selection,
            sort_x_by=sort_x_by,
            sort_y_by=sort_y_by,
            sort_x_order=sort_x_order,
            sort_y_order=sort_y_order,
            bins=bins,
        )

        def selection_callback(event):
            indices = plotter.source.selected.indices
            new_selection = plotter.plot_indices_to_selection(indices)
            self._selection_history[name] = new_selection

        # Merge the relevant 'plot_options' for this tab, into kwargs
        # (which will apply to all tabs)
        #
        # Thus a call to add_plot like this:
        #   session.add_plot(..., plot_options={tabname: {'y_axis_type': 'log'}})
        # would result in plotter.plot getting called as
        #   plotter.plot(y_axis_type='log')

        plot_options = kwargs.pop("plot_options", {})
        plot_options = plot_options.get(tabname, {})
        plot_options.setdefault("sizing_mode", "stretch_both")

        kwargs.update(plot_options)

        p = plotter.plot(**kwargs)
        p.on_event(SelectionGeometry, selection_callback)

        return p

    def add_plot(
        self,
        name,
        based_on=None,
        notebook=True,
        html_display_link=True,
        **kwargs,
    ):
        """Add a new plot

        Renders a set of interactive Bokeh plots in a tabbed layout
        (see the options below controlling how these are displayed).

        The plots have different titles, depending on the mode ('set
        mode' or 'missingness mode', see :py:class:`PlotSession`).
        The included plots are shown in the table below.  More detail
        about the plot can be found under the documentation for its
        plot class.

        Naming the plot allows any interactive selection made in the
        plot to be referred to later.

        ====================================  ============================  ============================================
        Set-mode plot                         Missingness-mode plot         Plot class
        ====================================  ============================  ============================================
        Set bar chart                         Value bar chart               :py:class:`SetBarChart`
        Intersection heatmap                  Combination heatmap           :py:class:`IntersectionHeatMap`
        Set cardinality histogram             Value count histogram         :py:class:`SetCardinalityHistogram`
        Intersection bar chart                Combination bar chart         :py:class:`IntersectionBarChart`
        Intersection cardinality histogram    Combination count histogram   :py:class:`IntersectionCardinalityHistogram`
        Intersection degree histogram         Combination length histogram  :py:class:`IntersectionDegreeHistogram`
        ====================================  ============================  ============================================

        Parameters
        ----------
        name : str
            The name of the plot.  This name is used to refer to any
            selection made in the plot.
        based_on : str or None
            The data to plot is taken from this selection (it is
            'based on' this selection).  Any selection made in *this*
            plot is a refinement of the `based_on` selection.
        notebook : bool
            Should the plot be displayed inline in the notebook?  A
            value of `False` starts and returns a Bokeh server for
            rendering the plots.
        html_display_link : bool
            Display an inline notebook link to the Bokeh server? Only
            used when `ok` is `False`, and when running in a
            notebook.
        **kwargs : dict
            Additional keyword arguments for each plot.

            Each keyword argument should be a dictionary, whose
            contents are used as keyword arguments for the `plot`
            method of the class for the corresponding plot.

            The arguments that are forwarded are listed below.  They
            have 'set mode' names (even in missingness mode).

            - `set_bar_chart`
            - `intersection_heatmap`
            - `set_cardinality_histogram`
            - `intersection_bar_chart`
            - `intersection_cardinality_histogram`
            - `intersection_degree_histogram`

            See the documentation for the individual plot classes for
            the accepted dictionary keys (any that are unknown are
            forwarded to :py:func:`bokeh.plotting.figure`).
        """

        ## Since this function starts a Bokeh server, stop various
        ## INFO and WARN messages being displayed in the notebook
        if self._verbose:
            logging.getLogger("bokeh.server").setLevel(logging.DEBUG)
            logging.getLogger("tornado").setLevel(logging.WARNING)
        else:
            logging.getLogger("bokeh.server").setLevel(logging.CRITICAL)
            logging.getLogger("tornado").setLevel(logging.CRITICAL)

        self._selection_history.new_selection(name, based_on)

        if name not in self._active_tabs:
            self._active_tabs[name] = 0

        def plot_app(doc):
            """Creates all the plot options and shows them in a tab layout"""
            p1 = self._add_subplot(
                SetBarChart, name, "set_bar_chart", **kwargs,
            )
            tab1 = Panel(
                child=p1,
                title="Set bar chart" if self._set_mode else "Value bar chart",
            )

            p2 = self._add_subplot(
                IntersectionHeatmap,
                name,
                "intersection_heatmap",
                **kwargs,
            )
            tab2 = Panel(
                child=p2,
                title="Intersection heatmap"
                if self._set_mode
                else "Combination heatmap",
            )

            p3 = self._add_subplot(
                IntersectionBarChart,
                name,
                "intersection_bar_chart",
                **kwargs,
            )
            tab3 = Panel(
                child=p3,
                title="Intersection bar chart"
                if self._set_mode
                else "Combination bar chart",
            )

            p4 = self._add_subplot(
                SetCardinalityHistogram,
                name,
                "set_cardinality_histogram",
                **kwargs,
            )
            tab4 = Panel(
                child=p4,
                title="Set cardinality histogram"
                if self._set_mode
                else "Value count histogram",
            )

            p5 = self._add_subplot(
                IntersectionCardinalityHistogram,
                name,
                "intersection_cardinality_histogram",
                **kwargs,
            )
            tab5 = Panel(
                child=p5,
                title="Intersection cardinality histogram"
                if self._set_mode
                else "Combination count histogram",
            )

            p6 = self._add_subplot(
                IntersectionDegreeHistogram,
                name,
                "intersection_degree_histogram",
                **kwargs,
            )
            tab6 = Panel(
                child=p6,
                title="Intersection degree histogram"
                if self._set_mode
                else "Combination length histogram",
            )

            tabs = Tabs(
                tabs=[tab1, tab2, tab3, tab4, tab5, tab6],
                active=self._active_tabs[name],
            )

            def active_tab_callback(attr, old, new):
                self._active_tabs[name] = new

            tabs.on_change("active", active_tab_callback)

            doc.add_root(tabs)
            doc.title = "setvis: " + name

        if self._verbose:
            logging.info(
                """

    # ********
    # To add a plot, insert a new cell below, type "add_plot(selected_indices)" and run cell.
    # ********"""
            )

        if notebook:
            show(plot_app)
        else:
            server = bokeh.server.server.Server({"/": plot_app}, port=0)
            server.start()
            from IPython.core.display import display, HTML

            if html_display_link:
                display(
                    HTML(
                        f"<a href='http://localhost:{server.port}' target='_blank' rel='noopener noreferrer'>"
                        f"Open plot in new browser tab</a> "
                    )
                )

            return server

    def add_selection(
        self,
        name,
        based_on=None,
        columns=None,
        intersections=None,
        records=None,
        invert=False,
    ):
        """Adds a selection to the plot session

        Parameters
        ----------
        name : str
            name of the selection
        based_on : str, optional
            name of the selection on which new selection is based, by default None
        columns : list, optional
            The included column names (may be any value returned by
            ``Membership.columns()``, which will generally be the same as
            in the underlying data source)
        records : list, optional
            The included record IDs (may be any value in
            ``Membership.columns()["_record_id"]``)
        intersections : list, optional
            The included intersection IDs (may be any value in
            ``Membership.intersections().index``)
        invert : bool, optional
            inverts selection, by default False
        """
        if not invert:
            parent = self._selection_history.membership(based_on)
            columns = parent.invert_column_selection(columns)
            intersections = parent.invert_intersection_selection(intersections)
            records = parent.invert_record_selection(records)

        selection = Selection(
            columns=columns, intersections=intersections, records=records
        )

        self._selection_history.new_selection(name, based_on)
        self._selection_history[name] = selection

    def selected_records(self, name=None, base_selection=None):
        """Returns the IDs of the records in the selection

        Parameters
        ----------
        name : str, optional
            name of the selection, by default None
        base_selection : _type_, optional
            name of the base selection from which selection is taken, by default None

        Returns
        -------
        pd.Series
            records IDs in selection
        """
        return self._selection_history.selected_records(name, base_selection)

    def membership(self, name=None):
        """Return the membership instance associated with the selection

        Parameters
        ----------
        name : str, optional
            the name of the selection for which to construct the ``Membership`` object,
            by default None

        Returns
        -------
        membership
            membership object associated with the selection
        """
        return self._selection_history.membership(name)

    def dict(self):
        """Returns a json-serializable dict representing the session

        It includes:
          - the plot selections (contained in _selection_history)
          - the active (currently-selected) tab in each Bokeh 'Tabs' layout

        It does not include any of the membership data itself

        This is used by .save() to save the session state to a file.
        """

        return {
            "selection_history": self._selection_history.dict(),
            "active_tabs": self._active_tabs,
        }

    def save(self, filename: str):
        """Saves the session state to a `json` file

        Parameters
        ----------
        filename : str
            name of the file used to save the session state
        """
        with open(filename, "w") as f:
            json.dump(self.dict(), f, cls=NPIntEncoder, indent=1)


class NPIntEncoder(json.JSONEncoder):
    """JSON encoder for numpy integer dtypes

    :meta private:
    """

    def default(self, obj):
        if np.issubdtype(obj, np.integer):
            return int(obj)
        return super().default(obj)
