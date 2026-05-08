Transformer Model
=================

``transformerModel`` is a PyTorch module for aggregating a target node
representation with its neighboring node representations. It is designed for
graph-style spatial data where each target observation is represented together
with sampled or multi-hop neighbors.

The model receives a sequence tensor whose first position is the target node and
whose remaining positions are neighbors. Transformer encoder layers refine the
whole sequence, and a final attention layer learns how strongly each neighbor
should contribute to the target representation.

Input Structure
---------------

The expected input shape is:

.. code-block:: text

   batched_data: Tensor[batch_size, hops + 1, hidden_dim]

The sequence dimension is organized as:

.. code-block:: text

   [target_node, neighbor_1, neighbor_2, ..., neighbor_hops]

The model stores ``seq_len`` as ``hops + 1``. The first token is treated as the
central node, and all remaining tokens are treated as neighbor representations.

Encoder Layers
--------------

The model contains ``n_layers`` repeated ``EncoderLayer`` blocks. Each block
contains:

* layer normalization before self-attention,
* multi-head self-attention,
* residual connection after attention,
* layer normalization before the feed-forward network,
* feed-forward transformation,
* residual connection after the feed-forward block.

These layers allow the target node and its neighbors to exchange information
before the final aggregation step.

Neighbor Attention
------------------

After the encoder stack, the model separates the output sequence into:

``node_tensor``
   The encoded target node representation.

``neighbor_tensor``
   The encoded neighbor representations.

The target representation is repeated along the neighbor dimension and
concatenated with each neighbor representation. A linear attention layer then
produces one score per neighbor. These scores are normalized with ``softmax`` so
that neighbors receive relative importance weights.

The weighted neighbor representations are summed into a single neighbor summary,
and the final output is computed as:

.. code-block:: text

   output = target_node_embedding + attention_weighted_neighbor_embedding

Parameters
----------

``hops``
   Number of neighbor positions represented in the input. The sequence length is
   ``hops + 1``.

``input_dim``
   Input feature dimension kept for compatibility with the training pipeline.
   In the current implementation, input features are expected to already match
   ``hidden_dim``.

``n_layers``
   Number of transformer encoder layers.

``num_heads``
   Number of attention heads in each encoder layer.

``hidden_dim``
   Hidden feature dimension used by self-attention, layer normalization, and the
   final target-neighbor aggregation.

``ffn_dim``
   Feed-forward dimension argument kept for API compatibility. The current
   implementation internally uses ``2 * hidden_dim``.

``dropout_rate``
   Dropout probability used after self-attention and feed-forward blocks.

``attention_dropout_rate``
   Dropout probability applied to attention weights inside multi-head
   self-attention.

Output
------

For batched input, the output shape is:

.. code-block:: text

   output: Tensor[batch_size, hidden_dim]

The output can be interpreted as the refined representation of the target node
after incorporating information from its neighbors.
