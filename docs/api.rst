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

.. autofunction:: SpaWeaver.utils.simonson_vHE

.. autofunction:: SpaWeaver.utils.HED_simulated_artifact

.. autofunction:: SpaWeaver.utils.re_features

.. autofunction:: SpaWeaver.utils.group_technical_effects

.. autofunction:: SpaWeaver.utils.remove_sinusoidal_component

Preprocessing functions
-----------------------

The following functions are available through ``SpaWeaver.pp``.

.. autofunction:: SpaWeaver.preprocess.read_VisiumHD

.. autofunction:: SpaWeaver.preprocess.load_he_emb

.. autofunction:: SpaWeaver.preprocess.mt_qc

.. autofunction:: SpaWeaver.preprocess.read_HE_image

.. autofunction:: SpaWeaver.preprocess.register_physical_to_pixel

.. autofunction:: SpaWeaver.preprocess.tiling_HE_patches

.. autofunction:: SpaWeaver.preprocess.extract_HE_patches_representation

.. autofunction:: SpaWeaver.preprocess.build_graph
