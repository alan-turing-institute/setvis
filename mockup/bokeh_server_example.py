import pandas as pd
from pathlib import Path

from bokeh.layouts import column
from bokeh.plotting import curdoc

import pace.plots

# filename = "Synthetic_APC_DIAG_Fields.csv"
# path = Path.cwd().parent / "examples" / "datasets" / filename
filename = "test_data_merged_10000.csv"
path = Path.cwd().parent.parent / "data" / filename

df = pd.read_csv(path, low_memory=False)

session = pace.plots.PlotSession(df, set_mode=False)
plotter_cls = pace.plots.SetBarChart
plotter = plotter_cls(session.missingness())
factor = 0.45
plotter.width = int(3840*factor)
plotter.height = int(2160*factor)
p = plotter.plot()

curdoc().add_root(column(p))