.. _installation-instructions:

Installation instructions
=========================

.. _python-env-setup:

Setting up the environment
--------------------------

Setvis places relatively few requirements on the system environment it
is installed in.  It is recommended to use a Python installation based
on a virtual environment or Conda.  This could be as part of an
existing project, or one set up specifically for running setvis by
following one of the configurations below.

Once you are happy with your environmental set-up, continue to
:ref:`installing-setvis`.


Conda
.....

This configuration has been tested with:

- `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_
  (based on Conda 4.10.3) with Python 3.8 on Windows 10 Pro 20H2

.. code:: bash

   # Create a new conda environment
   conda create -n paceenv python=3.8

   # Activate the newly-created conda environment
   conda activate paceenv

   # Install git if it is not available on your system.
   # This command is unnecessary if git is already installed.
   conda install git

   # Install the Bottleneck conda package and its dependencies
   conda install Bottleneck=1.3.2


Next, :ref:`install setvis <installing-setvis>`.


Python venv (virtual environment)
.................................

This configuration has been tested with:

- Python 3.8.2 on MacOS 10.15 (Catalina)

.. code:: bash

   # Create a virtual environment, in a new directory 'setvis-venv'
   python -m venv setvis-venv

   # Use the following command to activate the venv
   source setvis-venv/bin/activate

   # Upgrade pip once working inside
   pip install --upgrade pip


- `Python documentation on virtual environments <https://docs.python.org/3/tutorial/venv.html>`_


Next, :ref:`install setvis <installing-setvis>`.


.. _installing-setvis:

Installing setvis
-----------------

Once your environment is set up and activated, and follow one of the
sets of instructions below.

Installing from PyPI
....................

This will be the preferred method for most users when the package has been released on PyPI.

(TODO)


Installing from GitHub
......................

Clone the repository from GitHub and enter the directory so created:

.. code:: bash

   git clone https://github.com/alan-turing-institute/visualising-data-profiles

   cd visualising-data-profiles


Most users should then run

.. code:: bash

   pip install ".[extra]"

which will install setvis and most of the optional extra dependencies.


Alternatively, run

.. code:: bash

   pip install .

which will include setvis and a minimal set of dependencies.

The part of the package in square brackets above ('[extra]') is a pip
`dependency extra <https://peps.python.org/pep-0508/#extras>`_ for
selecting optional extra packages to install. See the :ref:`next
section <package-extras>` for a full list of these options, which can
be used to configure the setvis installation.


.. _package-extras:

Extras and fine-tuning the installation
.......................................

This section applies whatever the source of the installation (from
PyPI or GitHub).  Setvis supports several optional features that can be
installed by passing various extra dependency flags to pip.

For instance: ``pip install ".[notebook]"`` (which installs the
notebook dependencies).

- ``extra``: ``[extra]`` is the same as ``[notebook,doc,test]``
- ``all``: ``[all]`` is the same as
  ``[notebook,doc,test,performance-extras,db]`` (includes all of the
  below)

- ``notebook``: for the functionality required by the notebook examples
- ``doc``: sphinx and other libraries for building the documentation
- ``test``: pytest and other libraries for running the tests

The following dependencies place additional requirements on the
environment where the package is to be installed:

- ``performance-extras``: `numexpr
  <https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/>`_ and
  `Bottleneck <https://bottleneck.readthedocs.io/en/latest/>`_, for
  improving the performance of numerical computations. **Requires a C
  compiler**: see `Bottleneck requirements
  <https://bottleneck.readthedocs.io/en/latest/intro.html#install>`_

- ``db``: to support the database interface (currently just `psycopg2
  <https://www.psycopg.org/docs/>`_). **Requires an installation of
  PostgreSQL**.



Installing setvis with Poetry (developers)
----------------------------------------

This project uses `Poetry <https://python-poetry.org/>`_ for
dependency management and packaging.  To contribute to setvis
development, follow the instructions below to set up a virtual
environment containing setvis and its dependencies.  See the `poetry
documentation <https://python-poetry.org/docs/>`_ for how to use this
for dependency management.

.. code:: bash

   # Clone this repository
   git clone https://github.com/alan-turing-institute/visualising-data-profiles
   cd visualising-data-profiles

   # Install this project and its dependencies into a virtual environment
   poetry install

   # Activate the virtual environment
   poetry shell

