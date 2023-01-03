.. PACE documentation master file, created by
   sphinx-quickstart on Mon Nov 15 16:05:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PACE's documentation!
================================

PACE is a python package for exploring and visualizing data
missingness (that is, the presence, number and pattern of `missing
data <https://en.wikipedia.org/wiki/Missing_data>`_ in a dataset).

It can also be used to visualize :doc:`set membership <set_mode>`,
of which data missingness is a special case.

It is designed to work particularly well when used `interactively
<interactive_use.rst>`_ from a notebook, but can also be used
non-interactively.

At the moment, PACE can load data from `pandas
<https://pandas.pydata.org/>`_ dataframes, csv files, and also
supports a Postgres database backend.  It is designed with large
datasets in mind -- PACE may be able to load the missingness
information from a dataset even if the dataset itself does not fit
in memory.


Contents
========

.. toctree::
   :maxdepth: 2

   installation
   overview
   tutorials
   api_reference

Examples
========

Put a couple of simple examples here


Acknowledgements
================

The development of the PACE software was supported by funding from the
Engineering and Physical Sciences Research Council (EP/N013980/1;
EP/K503836/1) and the Alan Turing Institute (R-LEE-005 and a
fellowship awarded to R.A. Ruddle).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
