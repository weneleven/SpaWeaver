API
===

SpaWeaver model API.

This page documents the two core classes currently exposed from
``model.py``. The class names below match the real source code.

RBF
---

.. autoclass:: model.RBF
   :members: get_bandwidth, forward
   :special-members: __init__
   :show-inheritance:

transformerModel
----------------

.. autoclass:: model.transformerModel
   :members: forward
   :special-members: __init__
   :show-inheritance:
