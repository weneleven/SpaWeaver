API
===

SpaWeaver model API.

This page documents the two core classes currently exposed from
``SpaWeaver.model``. The class names below match the real source code in
``model.py``.

RBF
---

.. autoclass:: SpaWeaver.model.RBF
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: get_bandwidth

   .. automethod:: forward

transformerModel
----------------

.. autoclass:: SpaWeaver.model.transformerModel
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: forward
