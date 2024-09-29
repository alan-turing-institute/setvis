.. _setvis-documentation:

setvis documentation
====================

This section contains information useful for anyone planning to generate the documentation of setvis locally.

Generating documentation locally
--------------------------------

This documentation is hosted online on Read the Docs.

The package uses Sphinx to generate documentation.  It can be built from within the ``docs`` directory.  To do this, first ensure that the doc requirements are installed using:

.. code:: bash

  pip install .[doc]

For the complete installation instructions for setvis see :ref:`Installation <installation-instructions>`.
You should now be able to run ``make html`` inside the ``docs``
directory and see the output in ``docs/build/html``. The output
should not need any "server" and you should be
able to see the docs by opening the file ``docs/build/html/index.html`` in your web browser, which can be done from a file explorer, or alternatively, still within the ``docs`` directory, run

on Debian:

.. code:: bash

  xdg-open build/html/index.html

or on macOS:

.. code:: bash

  open build/html/index.html

