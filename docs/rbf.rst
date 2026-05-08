RBF Kernel
==========

``RBF`` is a PyTorch module that computes a multi-scale radial basis function
kernel matrix from an input feature matrix. It is mainly used as the kernel
component of ``MMD_loss`` to measure distribution differences between two sets
of embeddings.

The input to ``RBF.forward`` is a tensor with shape
``(n_samples, n_features)``. The module first computes all pairwise squared
Euclidean distances with ``torch.cdist``. It then converts these distances into
Gaussian kernel similarities and returns a matrix with shape
``(n_samples, n_samples)``. Each entry in this matrix represents the similarity
between two samples under the RBF kernel.

Multi-Scale Bandwidths
----------------------

Instead of using only one Gaussian bandwidth, ``RBF`` builds several bandwidth
scales. The number of scales is controlled by ``n_kernels`` and the spacing
between adjacent scales is controlled by ``mul_factor``.

For example, with the default ``n_kernels=5`` and ``mul_factor=2.0``, the module
evaluates multiple kernels centered around the base bandwidth and sums their
outputs. This makes the kernel less sensitive to a single bandwidth choice and
usually gives a more stable MMD signal.

Bandwidth Estimation
--------------------

If ``bandwidth`` is provided, the module uses it as a fixed bandwidth. If
``bandwidth`` is ``None``, the bandwidth is estimated from the current batch by
dividing the sum of the pairwise squared distance matrix by
``n_samples ** 2 - n_samples``. The diagonal distances are zero, so this is
equivalent to averaging over the off-diagonal sample pairs.

This means the kernel adapts to the scale of the embeddings in each batch. It is
convenient when embedding magnitudes vary across datasets or training stages,
but it also means the exact kernel values depend on the batch composition.

Parameters
----------

``n_kernels``
   Number of Gaussian kernels with different bandwidth scales. The default is
   ``5``.

``mul_factor``
   Multiplicative spacing factor between bandwidth scales. The default is
   ``2.0``.

``bandwidth``
   Fixed base bandwidth. If set to ``None``, the bandwidth is estimated from
   the input batch.

Input and Output
----------------

Input:

.. code-block:: text

   X: Tensor[n_samples, n_features]

Output:

.. code-block:: text

   K: Tensor[n_samples, n_samples]

Typical Usage
-------------

``RBF`` is used by ``MMD_loss`` in ``model.py``. ``MMD_loss`` stacks two
embedding sets, computes one shared kernel matrix with ``RBF``, and compares
within-domain and cross-domain similarities.
