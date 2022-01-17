#!/bin/bash

# Runs the notebooks in 'notebooks', apart from the indicated
# exceptions, but does not check their output (only that they ran
# successfully)

# Returns 0 if they all succeeded, nonzero otherwise (1 if there was
# an error when executing the notebook

cd ../notebooks

mkdir -p ../_notebook-output

find . -maxdepth 1 \
     -name '*.ipynb' \
     ! -name 'Postgres example.ipynb' \
     -print0 |
    xargs -0 -n 1 -I {} papermill "{}" "../_notebook-output/{}"
