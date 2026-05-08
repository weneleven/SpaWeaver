Installation
============

Clone the repository and install the dependencies used by your workflow.

.. code-block:: bash

   git clone https://github.com/<user-or-org>/SpaWeaver.git
   cd SpaWeaver

For local documentation builds, install the documentation requirements:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser to inspect the generated site.

Read the Docs
-------------

1. Push ``docs/`` and ``.readthedocs.yaml`` to GitHub.
2. Import the GitHub repository in Read the Docs.
3. Keep the default build command; RTD will read ``.readthedocs.yaml`` and build
   with ``docs/conf.py``.

If SpaWeaver is part of a larger repository, place ``.readthedocs.yaml`` at the
repository root and adjust ``sphinx.configuration`` if needed.
