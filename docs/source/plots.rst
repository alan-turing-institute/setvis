.. _plots:

Plotting and interactivity
==========================

The `setvis.plots` module
-------------------------

.. automodule:: setvis.plots
   :members:


.. _plot_outside_notebook:
Plotting outside of a notebook
------------------------------

.. note::

   This is 'advanced' use of Setvis, and is not as well tested as the notebook
   based workflow.


After creating a :class:`PlotSession` object, interactive plots can be
created with the :meth:`PlotSession.add_plot` method.  The usual,
Jupyter-notebook-based workflow, creates these inline in the notebook.

When calling :meth:`PlotSession.add_plot` with ``notebook=False``, a
Bokeh server is started and returned, enabling SetVis to be used outside of
a Jupyter notebook.

Note that this usage is still *interactive*, but the plots are no longer
confined to a notebook.  One use-case for this is to show the plots on a
large external display.

The code below creates and starts the Bokeh server.

.. code-block:: python

   import pandas as pd
   from setvis.plots import PlotSession

   ## This csv file can be found in the Setvis git repository
   df = pd.read_csv("examples/datasets/Synthetic_APC_DIAG_Fields.csv")

   session = PlotSession(df)

   ## Create a plot
   bokeh_plot_server = session.add_plot(name="Plot 3", notebook=False, html_display_link=False)

   ## Display the URL of the plot that was just created
   print(f"Connect to http://localhost:{bokeh_plot_server.port}/ to see the plot")

   ## Start and enter the event loop (this command blocks)
   ## Not required if running inside Jupyter
   bokeh_plot_server.run_until_shutdown()


The last command (``run_until_shutdown()``) is not required if
creating the plot from inside Jupyter (it will be attached to
Jupyter's event loop).  See the Bokeh documentation for more
information on using a Bokeh server, including how it can be embedded
in other applications.


.. seealso::

   - The example notebook in the Setvis repository `Example - plotting outside the notebook.ipynb <https://github.com/alan-turing-institute/setvis/blob/main/notebooks/Example%20-%20plotting%20outside%20the%20notebook.ipynb>`_ (GitHub link)
   - The Bokeh documentation on `Bokeh server APIs <https://docs.bokeh.org/en/latest/docs/user_guide/server/library.html>`_
