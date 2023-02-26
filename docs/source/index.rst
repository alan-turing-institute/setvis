.. setvis documentation master file, created by
   sphinx-quickstart on Mon Nov 15 16:05:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Setvis Documentation Index
==========================

Setvis is a python package for exploring and visualizing data
missingness (that is, the presence, number and pattern of `missing
data <https://en.wikipedia.org/wiki/Missing_data>`_ in a dataset).

It can also be used to visualize set membership of which data
missingness is a special case.

It is designed to work particularly well when used interactively from
a notebook, but can also be used non-interactively.

At the moment, setvis can load data from `pandas
<https://pandas.pydata.org/>`_ dataframes, csv files, and also
supports a Postgres database backend.  It is designed with large
datasets in mind -- setvis may be able to load the missingness
information from a dataset even if the dataset itself does not fit
in memory.


Contents
========

.. toctree::
   :maxdepth: 2

   installation
   tutorials
   documentation
   api_reference

Examples
========

Put a couple of simple examples here


Acknowledgements
================

The development of the setvis software was supported by funding from the
Engineering and Physical Sciences Research Council (EP/N013980/1;
EP/K503836/1) and the Alan Turing Institute.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
