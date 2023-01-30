# setvis [![Python Package](https://github.com/alan-turing-institute/setvis/actions/workflows/main.yml/badge.svg)](https://github.com/alan-turing-institute/setvis/actions/workflows/main.yml)

A tool for visualising patterns of missingness in data. Setvis is matrix-based set visualization that operates with datasets using a memory-efficient architecture.

## Installation

### Pip

#### Installing the module

These instructions have been tested with:
- Python 3.8.2 on MacOS 10.15 (Catalina)

```
git clone https://github.com/alan-turing-institute/setvis

cd setvis

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
```

Then run either:

```
pip install ".[extra]"
```

which will install setvis and most of the optional extra dependencies

or run:

```
pip install .
```

which will include just setvis and a minimal set of dependencies.


#### Extras and fine-tuning the installation

There are several dependency flags that can be passed to pip to install
various optional dependencies.  For instance: `pip install ".[notebook]"` (which installs the notebook dependencies).

 - `extra`: `[extra]` is the same as `[notebook,doc,test]`
 - `all`: `[all]` is the same as `[notebook,doc,test,performance-extras,db]` (includes all of the below)

 - `notebook`: for the functionality required by the notebook examples
 - `doc`: sphinx and other libraries for building the documentation
 - `test`: pytest and other libraries for running the tests

The following dependencies place additional requirements on the environment where the package is to be installed:
 - `performance-extras`: [numexpr](https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/) and [Bottleneck](https://bottleneck.readthedocs.io/en/latest/), for improving the performance of numerical computations. **Requires a C compiler**: see [Bottleneck requirements](https://bottleneck.readthedocs.io/en/latest/intro.html#install)
 - `db`: to support the database interface (currently just [psycopg2](https://www.psycopg.org/docs/)). **Requires an installation of PostgreSQL**.


#### Running the tutorial notebooks

The Bokeh plots produced by setvis require the package `notebook >= 6.4` to display properly.

Installing the `notebook` extra dependency set (see above) will include everything
required to run setvis in a notebook, and to run the tutorial examples
that do not need a database connection. For the latter, install `db`
as well.

If the installation succeeded, it should be possible to run the
notebooks in the `notebooks` directory of the repository:

```
python -m jupyter notebook
```

### Conda

These instructions have been tested with:
- [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) (based on Conda 4.10.3) with Python 3.8 on Windows 10 Pro 20H2

```posh
# Create an activate a conda environment
conda create -n paceenv python=3.8
conda activate paceenv

# This is unnecessary if git is already installed
conda install git

# Clone this repository
git clone https://github.com/alan-turing-institute/setvis
cd setvis

# Install the Bottleneck conda package and its dependencies
conda install Bottleneck=1.3.2

# Install setvis itself and the remaining dependencies with pip
pip install ".[all]"
```

If the commands above succeed, it should be possible to run the notebooks in `notebooks`, with

```
jupyter notebook notebooks
```

### Poetry

```
# Clone this repository
git clone https://github.com/alan-turing-institute/setvis
cd setvis

# Install this project and its dependencies into a virtual environment
poetry install

# Activate the virtual environment
poetry shell
```

## Acknowledgements

The development of the setvis software was supported by funding from the Engineering and Physical Sciences Research Council (EP/N013980/1; EP/K503836/1) and the Alan Turing Institute.
