# Visualising Data Profiles
A tool for visualising patterns of missingness in data

## Installation

### Pip

These instructions have been tested with:
- Python 3.8.2 on MacOS 10.15 (Catalina)

```
git clone https://github.com/alan-turing-institute/visualising-data-profiles

cd visualising-data-profiles

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install .
```

If the commands above succeed, it should be possible to run the notebooks in `notebooks`, with

```
jupyter notebook notebooks
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
pip install .
```

If the commands above succeed, it should be possible to run the notebooks in `notebooks`, with

```
jupyter notebook notebooks
```

## Acknowledgements

The development of the PACE software was supported by funding from the Engineering and Physical Sciences Research Council (EP/N013980/1; EP/K503836/1) and the Alan Turing Institute (R-LEE-005 and a fellowship awarded to R.A. Ruddle).
