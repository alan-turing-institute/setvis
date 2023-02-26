#!/bin/bash

# Runs the notebooks in 'notebooks', apart from the indicated
# exceptions, but does not check their output (only that they ran
# successfully)

# Returns zero if they all succeeded, nonzero otherwise

cd ../notebooks

find . -maxdepth 1 \
     -name '*.ipynb' \
     ! -name 'Tutorial 3 - Loading data from Postgres.ipynb' \
     ! -name 'Tutorial 3 (supplemental) - Create the Postgres database.ipynb' \
     -print0 |
    xargs -0 -n 1 -I {} jupyter execute "{}"
