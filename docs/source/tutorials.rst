Tutorials
=========

The PACE repository includes some tutorials and extended examples in
the form of Jupyter notebooks.  The tutorials are designed to
introduce PACE concepts and be run interactively.

This section describes how to install and run these.

The source for all notebooks can be found `within the PACE GitHub
repository
<https://github.com/alan-turing-institute/visualising-data-profiles/tree/main/notebooks>`_.


Install and run the notebooks
-----------------------------

To access the tutorials, after setting up a suitable python
environment (see the :ref:`relevant section of the installation
instructions <python-env-setup>`), clone and install PACE from GitHub
source.

.. code:: bash

   git clone https://github.com/alan-turing-institute/visualising-data-profiles

   cd visualising-data-profiles


Then run the command below to install the ``notebook`` extra dependency
set, which will include everything required to run PACE in a notebook,
and to run the tutorial examples that do not need a database
connection.

.. code:: bash

   pip install ".[notebook]"


To be able to run the postgres examples, install the ``db`` extra
dependency set as well (see :ref:`Extras and fine-tuning the
installation <package-extras>`).

If the installation succeeded, it should be possible to run the
notebooks in the ``notebooks`` directory of the repository:

.. code:: bash

   cd notebooks

   python -m jupyter notebook

.. warning::

   The Bokeh plots produced by PACE require the package ``notebook``
   with version **>= 6.4** in order to display properly.  If PACE is
   installed as described above, this will be included automatically.
   If using a 'minimal' PACE installation alongside an existing
   notebook installation, please ensure that this requirement is met.
