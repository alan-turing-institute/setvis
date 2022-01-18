# Visualising Data Profiles
A tool for visualising patterns of missingness in data

## Installation

### Pip

#### Installing the module

These instructions have been tested with:
- Python 3.8.2 on MacOS 10.15 (Catalina)

```
git clone https://github.com/alan-turing-institute/visualising-data-profiles

cd visualising-data-profiles

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
```

Then run either:

```
pip install ".[extra]"
```
which will install PACE and most of the optional extra dependencies

or run:

```
pip install .
```

which will include just PACE and a minimal set of dependencies.


#### Extras and fine-tuning the installation

There are several dependency flags that can be passed to pip to install
various optional dependencies.  For instance: `pip install .[notebook]` (which installs the notebook dependencies).

 - `extra`: `[extra]` is the same as `[notebook,doc,test]`
 - `all`: includes all of the below

 - `notebook`: for the functionality required by the notebook examples
 - `doc`: sphinx and other libraries for building the documentation
 - `test`: pytest and other libraries for running the tests

The following dependencies have additional environmental dependencies:
 - `performance-extras`: [numexpr](https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/) and [Bottleneck](https://bottleneck.readthedocs.io/en/latest/), for improving the performance of numerical computations. **Requires a C compiler:** See [https://bottleneck.readthedocs.io/en/latest/intro.html#install](Bottleneck requirements)
 - `db`: to support the database interface (currently just [psycopg2](https://www.psycopg.org/docs/)). **Requires an installation of PostgreSQL**.


#### Running the tutorial notebooks

The bokeh plots require `notebook >= 6.4` to work properly.

Installing the `notebook` extra dependency will include everything
required to run pace in a notebook, and to run the tutorial examples
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
git clone https://github.com/alan-turing-institute/visualising-data-profiles
cd visualising-data-profiles
 
# Install the Bottleneck conda package and its dependencies
conda install Bottleneck=1.3.2
 
# Install pace itself and the remaining dependencies with pip
pip install ".[extra]"
```

If the commands above succeed, it should be possible to run the notebooks in `notebooks`, with

```
jupyter notebook notebooks
```

### Poetry

```
# Clone this repository
git clone https://github.com/alan-turing-institute/visualising-data-profiles
cd visualising-data-profiles

# Install this project and its dependencies into a virtual environment
poetry install

# Activate the virtual environment
poetry shell
```

## Acknowledgements

The development of the PACE software was supported by funding from the Engineering and Physical Sciences Research Council (EP/N013980/1; EP/K503836/1) and the Alan Turing Institute (R-LEE-005 and a fellowship awarded to R.A. Ruddle).
