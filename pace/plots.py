from .datastructure import MissingData
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LinearColorMapper, tools, CustomJS
from bokeh.palettes import Oranges256
from bokeh.transform import transform, linear_cmap
from bokeh.models.widgets import Panel, Tabs  # not sure if needed
from bokeh.io import output_file, show  # not sure if needed
import pandas as pd
import numpy as np
import logging

import base64
from IPython.display import Javascript, display
from ipywidgets import widgets

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TOOLS = "box_select, tap, reset"


class PlotMissingData(MissingData):
    """
    Class for plotting missing data 
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.combinations_matrix = self.dict_to_matrix()
        self.long_combinations_matrix = self.dict_to_long_matrix()

    def dict_to_long_matrix(self):
        """
        Takes the dictionary of the missing_data instance that holds the
        information about the missing combinations and converts it into
        the matrix/column data source format required for the plots.

        difference to dict_to_matrix: not full matrix
        """
        count = 0  # counts over number of combination
        index = []
        variable = []
        value = []
        for combination, rows in self.missing_combinations.items():
            index += list(
                np.ones((len(combination), 1), dtype=int)[:, 0] * count
            )
            variable += list(
                combination
            )  # should combinations be list instead of tuple all along?
            value += list(
                np.ones((len(combination), 1), dtype=int)[:, 0] * len(rows)
            )
            count += 1
        zipped_list = list(zip(index, variable, value))
        long_combination_matrix = pd.DataFrame(
            zipped_list, columns=["index", "variable", "value"]
        )
        return long_combination_matrix

    def dict_to_matrix(self):
        """
        Takes the dictionary of the missing_data instance that holds the
        information about the missing combinations and converts it into
        the matrix/column data source format required for the plots.
        """
        fields = list(
            self.missing_fields["Field"][self.missing_fields["Field"] != 0]
        )
        # fields = list()
        combinations_matrix = pd.DataFrame(
            0, index=np.arange(len(self.missing_combinations)), columns=fields
        )

        index = 0
        for combination, rows in self.missing_combinations.items():
            count = len(rows)
            for field_name in combination:
                combinations_matrix.loc[
                    index, field_name
                ] = count  # TODO: at instead count?
            index += 1

        return combinations_matrix

    def value_bar_chart(self, selected_indices=[]):
        """Plots the number of missing values for each field in the dataset as
        a bar chart :param missing_data: MissingDataInstance

        """
        source = ColumnDataSource(self.missing_fields)
        width = 0.5  # width of bars
        p = figure(title="Value bar chart", tools=TOOLS, width=960, height=960)
        p.vbar(x="Field", top="Count", width=width, source=source)
        p.xaxis.ticker = list(self.missing_fields["Field"])
        p.xaxis.major_label_overrides = self.lookup_table_backward
        p.xaxis.major_label_orientation = "vertical"
        p.yaxis.axis_label = "Number of missing values"

        # make a custom javascript callback that exports the indices of the selected points
        # to the Jupyter notebook
        callback = CustomJS(
            args=dict(source=source),
            code="""
                                console.log('Running CustomJS callback now.');
                                var indices = source.selected.indices;
                                var data = source.selected.data;
                                var kernel = IPython.notebook.kernel;
                                kernel.execute("selected_indices = " + indices)
                                """,
        )

        # set the callback to run when a selection geometry event occurs in the figure
        p.js_on_event("selectiongeometry", callback)
        selected_indices = list(source.data["index"])
        return p

    def combination_heatmap(self, selected_indices=[]):
        """Plots a heatmap with all unique combinations of missing fields
        :param missing_data: MissingDataInstance

        """
        source = ColumnDataSource(self.long_combinations_matrix)

        p = figure(
            title="Combination heatmap",
            width=960,
            height=960,
            tools=TOOLS,  #
            y_range=[
                str(x)
                for x in np.arange(
                    self.long_combinations_matrix["index"].max() + 1
                )
            ],
        )

        p.background_fill_color = "#cccccc"
        p.grid.visible = False

        p.rect(
            y="index",
            x="variable",
            source=source,
            width=1.0,
            height=1.0,
            fill_color=linear_cmap(
                field_name="value",
                palette=list(reversed(Oranges256)),
                low=1,
                high=self.long_combinations_matrix["value"].max(),
                low_color="#00000000",
            ),
            line_color="#cccccc",
        )
        p.xaxis.major_label_orientation = "vertical"
        p.yaxis.axis_label = "Combinations"
        p.xaxis.axis_label = "Fields"
        p.yaxis.major_label_text_color = None

        callback = CustomJS(
            args=dict(source=source),
            code="""
                                console.log('Running CustomJS callback now.');
                                var indices = source.selected.indices;
                                var data = source.selected.data;
                                var kernel = IPython.notebook.kernel;
                                kernel.execute("selected_indices = " + indices)
                                """,
        )

        # set the callback to run when a selection geometry event occurs in the figure
        p.js_on_event("selectiongeometry", callback)
        selected_indices = list(source.data["index"])
        return p

    def plot_data(self, selected_indices=[]):
        """Creates all the plot options and shows them in a tab layout"""
        p1 = self.value_bar_chart(selected_indices=[])
        tab1 = Panel(child=p1, title="Value bar chart")
        p2 = self.combination_heatmap()
        tab2 = Panel(child=p2, title="Combination heatmap")
        tabs = Tabs(tabs=[tab1, tab2])

        show(tabs)
        logging.info(
            f"""
            
    # ********
    # To add a plot, insert a new cell below, type "add_plot(selected_indices)" and run cell.
    # ********"""
        )
