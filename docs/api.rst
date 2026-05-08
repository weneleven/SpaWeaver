API
===

SpaWeaver model classes.

This page documents the two core classes currently exposed from
``SpaWeaver.model``:

``RBF``
   Computes a multi-scale radial basis function kernel matrix. It is used by
   ``MMD_loss`` to compare embedding distributions.

``transformerModel``
   Encodes a target node together with neighboring representations and returns
   an attention-weighted target representation.

Note
----

This module imports PyTorch components from ``torch``, ``torch.nn``, and
``torch.nn.functional``. ``RBF`` is also used by ``MMD_loss`` in the same
module.

RBF
---

``class SpaWeaver.model.RBF(n_kernels=5, mul_factor=2.0, bandwidth=None)``

Bases: ``torch.nn.Module``

Multi-scale radial basis function kernel module.

``RBF`` receives a feature matrix and returns a sample-by-sample kernel
similarity matrix. Internally, it computes pairwise squared Euclidean distances
with ``torch.cdist(X, X) ** 2`` and converts those distances into Gaussian RBF
similarities. Multiple bandwidth scales are evaluated and summed, making the
kernel less dependent on a single bandwidth choice.

Attributes
----------

``bandwidth_multipliers``
   Tensor of bandwidth scale multipliers generated from ``n_kernels`` and
   ``mul_factor``.

``bandwidth``
   Fixed base bandwidth. If ``None``, the base bandwidth is estimated from the
   current input batch.

__init__(n_kernels=5, mul_factor=2.0, bandwidth=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize the multi-scale RBF kernel.

Parameters
----------

``n_kernels`` : int, default=5
   Number of Gaussian kernels with different bandwidth scales.

``mul_factor`` : float, default=2.0
   Multiplicative factor used to space adjacent bandwidth scales.

``bandwidth`` : float or None, default=None
   Fixed base bandwidth. When ``None``, the bandwidth is estimated during
   ``forward`` from the pairwise squared distance matrix.

get_bandwidth(L2_distances)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the bandwidth used by the kernel computation.

If ``self.bandwidth`` is set, this method returns the fixed value directly. If
``self.bandwidth`` is ``None``, it estimates the bandwidth as:

.. code-block:: text

   L2_distances.sum() / (n_samples ** 2 - n_samples)

The diagonal entries of ``L2_distances`` are zero, so this is equivalent to
averaging over off-diagonal sample pairs.

Parameters
----------

``L2_distances`` : torch.Tensor
   Pairwise squared Euclidean distance matrix with shape
   ``(n_samples, n_samples)``.

Returns
-------

float or torch.Tensor
   Fixed or batch-estimated base bandwidth.

forward(X)
~~~~~~~~~~

Compute the summed multi-scale RBF kernel matrix.

The method moves ``bandwidth_multipliers`` to the same device as ``X``, computes
pairwise squared distances, evaluates the Gaussian kernels at multiple
bandwidth scales, and sums the resulting kernel matrices over the scale
dimension.

Parameters
----------

``X`` : torch.Tensor
   Input feature matrix with shape ``(n_samples, n_features)``.

Returns
-------

torch.Tensor
   Kernel matrix with shape ``(n_samples, n_samples)``.

transformerModel
----------------

``class SpaWeaver.model.transformerModel(hops, input_dim, n_layers=6, num_heads=8, hidden_dim=64, ffn_dim=64, dropout_rate=0.0, attention_dropout_rate=0.1)``

Bases: ``torch.nn.Module``

Transformer encoder for aggregating a target node with neighboring
representations.

The input is expected to be a sequence tensor where the first position is the
target node and the remaining positions are neighbors. The model applies a stack
of ``EncoderLayer`` blocks, then learns attention weights over the neighbor
tokens conditioned on the encoded target token. The returned representation is
the sum of the encoded target token and the attention-weighted neighbor summary.

Attributes
----------

``seq_len``
   Sequence length used by the model, set to ``hops + 1``.

``layers``
   ``torch.nn.ModuleList`` containing ``n_layers`` encoder blocks.

``final_ln``
   Final layer normalization applied after the encoder stack.

``attn_layer``
   Linear layer mapping concatenated target-neighbor features of size
   ``2 * hidden_dim`` to one attention score per neighbor.

``scaling``
   Learnable scalar parameter initialized to ``0.5``. It is defined in the
   implementation but is not used in ``forward``.

__init__(hops, input_dim, n_layers=6, num_heads=8, hidden_dim=64, ffn_dim=64, dropout_rate=0.0, attention_dropout_rate=0.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize the transformer aggregation model.

Parameters
----------

``hops`` : int
   Number of neighbor positions represented in the input. The model sets
   ``seq_len = hops + 1``.

``input_dim`` : int
   Input feature dimension kept as an attribute for compatibility with the
   training pipeline. The current implementation expects the last dimension of
   ``batched_data`` to match ``hidden_dim``.

``n_layers`` : int, default=6
   Number of encoder layers.

``num_heads`` : int, default=8
   Number of attention heads in each ``EncoderLayer``.

``hidden_dim`` : int, default=64
   Hidden dimension used by attention, feed-forward layers, normalization, and
   target-neighbor aggregation.

``ffn_dim`` : int, default=64
   Constructor argument kept for API compatibility. The implementation sets
   ``self.ffn_dim = 2 * hidden_dim`` regardless of this argument.

``dropout_rate`` : float, default=0.0
   Dropout probability used after self-attention and feed-forward blocks.

``attention_dropout_rate`` : float, default=0.1
   Dropout probability used inside multi-head attention.

forward(batched_data)
~~~~~~~~~~~~~~~~~~~~~

Encode the target-neighbor sequence and return an aggregated target
representation.

The method performs the following steps:

1. Pass ``batched_data`` through each encoder layer in ``self.layers``.
2. Apply ``final_ln`` to the encoded sequence.
3. Split the sequence into the first token, ``node_tensor``, and the remaining
   tokens, ``neighbor_tensor``.
4. Repeat the target token along the neighbor dimension and concatenate it with
   each neighbor token.
5. Use ``attn_layer`` followed by ``softmax`` to compute one normalized weight
   per neighbor.
6. Sum the weighted neighbor representations and add the result to the encoded
   target representation.
7. Return the result after ``squeeze()``.

Parameters
----------

``batched_data`` : torch.Tensor
   Input tensor with shape ``(batch_size, hops + 1, hidden_dim)``. The first
   sequence position is interpreted as the target node, and the remaining
   positions are interpreted as neighbors.

Returns
-------

torch.Tensor
   Aggregated target representation. For typical batched input with
   ``batch_size > 1``, the shape is ``(batch_size, hidden_dim)``. Because the
   implementation calls ``squeeze()``, singleton dimensions can be removed when
   ``batch_size`` is ``1``.
