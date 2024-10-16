# setvis

[![Python Package](https://github.com/alan-turing-institute/setvis/actions/workflows/main.yml/badge.svg)](https://github.com/alan-turing-institute/setvis/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/setvis/badge/?version=latest)](https://setvis.readthedocs.io/en/latest/?badge=latest)

Setvis is a python library for visualising set membership and patterns of missingness in data.

The plotting and interactive workflow of Setvis is designed for use within a Jupyter notebook (although it is possible to run outside Jupyter). The other components of Setvis can be used interactively or programmatically. The interactive plots are powered by [Bokeh](https://docs.bokeh.org/en/latest/index.html) widgets.

It operates on data using a memory efficient architecture, and supports loading data from flat files, Pandas dataframes, and directly from a Postgres database.

## Documentation

[The setvis documentation](https://setvis.readthedocs.io/en/latest/index.html) is hosted on Read the Docs.

## Installation (quick start)

**For the complete installation instructions, consult [the installation page of the documentation](https://setvis.readthedocs.io/en/latest/installation.html), which includes information on some extra installation options and setting up a suitable environment on several platforms.**

We recommend installing setvis in a python virtual environment or Conda environment.

To install setvis, most users should run:

```
pip install 'setvis[notebooks]'
```

This will include everything to run setvis in a notebook, and to run the tutorial examples that do not need a database connection.

The Bokeh plots produced by setvis require the package `notebook >= 6.4` to display properly.  This will be included when installing setvis using the command above.


## Tutorials

For basic examples, please see the two example notebooks:
- [Missingness example](https://github.com/alan-turing-institute/setvis/blob/main/notebooks/Example%20-%20import%20data%20to%20visualize%20missingness.ipynb)
- [Set example](https://github.com/alan-turing-institute/setvis/blob/main/notebooks/Example%20-%20import%20data%20to%20visualize%20sets.ipynb)

Additionally, there is a series of Tutorials notebooks, starting with [Tutorial 1](https://github.com/alan-turing-institute/setvis/blob/main/notebooks/Tutorial%201%20-%20Overview%20and%20an%20example%20analysis.ipynb).

After installing setvis, to follow theses tutorials interactively you will need to clone or download this repository. Then start jupyter from within it:

```
python -m jupyter notebook notebooks
```

## Notice

The setvis software is released under the Apache Licence, version 2.0. See [LICENCE](./LICENCE) for details.

The data files [`./examples/datasets/simpsons - Format 1.csv`](https://github.com/alan-turing-institute/setvis/blob/main/examples/datasets/simpsons%20-%20Format%201.csv) and [`./examples/datasets/simpsons - Format 2.csv`](https://github.com/alan-turing-institute/setvis/blob/main/examples/datasets/simpsons%20-%20Format%202.csv), are based on a data file included in [UpSet](https://github.com/VCG/upset), copyright Visual Computing Group, Harvard, and distributed here under the terms of the MIT Licence.

The other data files in `./examples/datasets/` are released under the [Creative Commons Attribution 4.0 International Licence (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).


## Acknowledgements

The development of the setvis software was supported by funding from the Engineering and Physical Sciences Research Council (EP/N013980/1; EP/R511717/1) and the Alan Turing Institute.
