Documentation
=============

For installation instructions see installation.

Generating Docs
===============

The package uses Sphinx to generate documentation.
In order to use the `Makefile` which is located
in the `docs` directory you will need (on a Debian Linux):

1. `apt-get install python3-sphinx`
2. `pip install nbsphinx`
3. `pip install pydata_sphinx_theme`

You should now be able to run `make html` inside the `docs`
directory and see the output in `docs/build/html`. The output
should not need any "server" and you should be
able to see the docs by opening the file ``docs/build/html/index.html`` in your web browser, which can be done from a file explorer, or alternatively, still within the ``docs`` directory, run

on Debian:

.. code:: bash
  xdg-open build/html/index.html
            
or on macOS:

.. code:: bash

  open build/html/index.html

