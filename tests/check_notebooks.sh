#!/bin/bash

# Runs the notebooks in 'notebooks', apart from the indicated
# exceptions, but does not check their output (only that they ran
# successfully)

# Returns zero if they all succeeded, nonzero otherwise

cd ../notebooks

mkdir -p ../_notebook-output

find . -maxdepth 1 \
     -name '*.ipynb' \
     ! -name 'Postgres example.ipynb' \
     -print0 |
    xargs -0 -n 1 -I {} papermill "{}" "../_notebook-output/{}"