Quickstart
==========

Read a dataset and run common preprocessing steps:

.. code-block:: python

   import SpaWeaver as sw

   adata = sw.pp.read_h5ad("sample.h5ad")
   adata = sw.pp.quality_control(adata, platform="VisiumHD")
   adata = sw.pp.preprocess_adata(adata, target_sum=1e4, n_hvg=3000)

Build a spatial graph from coordinates:

.. code-block:: python

   adj = sw.pp.build_graph(
       adata.obsm["spatial"],
       graph_type="knn",
       num_neighbors=7,
       weighted=False,
       symmetric=True,
   )

Use the model module:

.. code-block:: python

   from SpaWeaver.model import transformerModel

   model = transformerModel(
       hops=3,
       input_dim=128,
       n_layers=1,
       num_heads=2,
       hidden_dim=128,
   )
