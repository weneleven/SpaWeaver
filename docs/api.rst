API
===

SpaWeaver API.

This page documents the core model classes and commonly used utility
functions.

RBF
---

.. autoclass:: SpaWeaver.model.RBF
   :members: get_bandwidth, forward
   :show-inheritance:

transformerModel
----------------

.. autoclass:: SpaWeaver.model.transformerModel
   :members: forward
   :show-inheritance:

Utility functions
-----------------

The following functions are available through ``SpaWeaver.utils``.

.. autofunction:: SpaWeaver.utils.compute_metrics

.. autofunction:: SpaWeaver.utils.compute_MoransI

.. autofunction:: SpaWeaver.utils.create_ImageEncoder
