Layers
======

The layer module provides custom neural network layers for time series models. These layers are utility components used by the forecasting models to handle specific operations needed for time series processing.

**Purpose:**

These layers are building blocks used internally by the forecasting models. While you typically don't need to use them directly, understanding them can help when customizing models.

**Common Use Cases:**

- **Dimension Manipulation**: Adding or permuting dimensions for compatibility
- **Causal Convolutions**: Ensuring predictions don't use future information
- **Positional Information**: Adding temporal context to transformer models

.. automodule:: deep_time_series.layer
   :members:
   :undoc-members:
   :show-inheritance:

Unsqueesze
~~~~~~~~~~

A layer that adds a dimension to the input tensor at the specified position.

**Purpose:**

Wraps PyTorch's ``unsqueeze()`` operation as a layer, useful for adding dimensions in sequential models.

**Initialization Parameters:**

- ``dim`` (int): The position at which to insert the new dimension. Must be between ``-input.ndim-1`` and ``input.ndim`` (inclusive).

**Input/Output:**

- **Input**: Any tensor with shape ``(...,)``
- **Output**: Tensor with an additional dimension at position ``dim``, shape ``(..., 1, ...)``

**When to Use:**

- Need to add a time dimension for sequential processing
- Converting between different tensor shapes in a model
- Making tensors compatible with layers expecting specific dimensions

**Example:**

.. code-block:: python

   import torch
   from deep_time_series.layer import Unsqueesze

   layer = Unsqueesze(dim=1)
   x = torch.randn(32, 64)  # (batch_size, features)
   y = layer(x)  # (32, 1, 64) - adds dimension at position 1

**Note:**

This is a simple wrapper around ``torch.unsqueeze()``. The name "Unsqueesze" is a typo in the original code but kept for backward compatibility.

.. autoclass:: deep_time_series.layer.Unsqueesze
   :members:
   :undoc-members:
   :show-inheritance:

Permute
~~~~~~~

A layer that permutes the dimensions of the input tensor according to the specified order.

**Purpose:**

Wraps PyTorch's ``permute()`` operation as a layer, useful for reordering dimensions.

**Initialization Parameters:**

- ``*dims`` (int): Variable number of integers specifying the desired dimension order. The number of arguments must match the input tensor's number of dimensions. Each integer represents the dimension index in the original tensor.

**Input/Output:**

- **Input**: Tensor with shape ``(d0, d1, ..., dn)``
- **Output**: Tensor with permuted dimensions according to ``dims``. If ``dims = (i0, i1, ..., in)``, output shape is ``(d[i0], d[i1], ..., d[in])``

**When to Use:**

- Converting between ``(batch, time, features)`` and ``(batch, features, time)`` formats
- Making tensors compatible with layers expecting different dimension orders
- Used in CNN models where convolutions expect ``(B, C, L)`` format

**Example:**

.. code-block:: python

   import torch
   from deep_time_series.layer import Permute

   layer = Permute(0, 2, 1)  # Swap dimensions 1 and 2
   x = torch.randn(32, 10, 64)  # (batch, time, features)
   y = layer(x)  # (32, 64, 10) - (batch, features, time)

**Common Patterns:**

- ``Permute(0, 2, 1)``: Convert ``(B, L, F)`` to ``(B, F, L)`` for Conv1d
- ``Permute(0, 2, 1)`` then back: Convert back after convolution

.. autoclass:: deep_time_series.layer.Permute
   :members:
   :undoc-members:
   :show-inheritance:

LeftPadding1D
~~~~~~~~~~~~~

A layer that adds left padding to 1D sequences. Useful for causal convolutions in time series models.

**Purpose:**

Ensures causal (non-leaking) convolutions by padding on the left side. This prevents the convolution from using future information when processing time series data.

**Initialization Parameters:**

- ``padding_size`` (int): Number of zero-padding elements to add at the beginning of the sequence. Must be non-negative.

**Input/Output:**

- **Input**: Tensor with shape ``(batch_size, sequence_length, n_features)``
- **Output**: Tensor with shape ``(batch_size, sequence_length + padding_size, n_features)``. The first ``padding_size`` time steps are zeros, followed by the original sequence.

**When to Use:**

- Implementing causal convolutions (e.g., in DilatedCNN)
- Ensuring temporal order is preserved
- Preventing information leakage from future to past

**How It Works:**

Adds zeros to the left (beginning) of the sequence, shifting the original data to the right. This allows convolutions to process the sequence while maintaining causality. The padding is created on the same device and with the same dtype as the input tensor.

**Example:**

.. code-block:: python

   import torch
   from deep_time_series.layer import LeftPadding1D

   layer = LeftPadding1D(padding_size=2)
   x = torch.randn(32, 10, 64)  # (batch, time, features)
   y = layer(x)  # (32, 12, 64) - 2 zeros added at the beginning

**Use in Dilated CNN:**

In dilated convolutions, the padding size is calculated as ``dilation * (kernel_size - 1)`` to ensure the output length matches the input length while maintaining causality.

**Important:**

- Only pads on the left (beginning)
- Padding values are zeros
- Maintains the same device and dtype as input

.. autoclass:: deep_time_series.layer.LeftPadding1D
   :members:
   :undoc-members:
   :show-inheritance:

PositionalEncoding
~~~~~~~~~~~~~~~~~~

Positional encoding layer for transformer models. Adds positional information to input embeddings.

**Purpose:**

Since transformers don't have inherent notion of sequence order, positional encoding adds temporal information to help the model understand the position of each time step.

**Initialization Parameters:**

- ``d_model`` (int): Embedding dimension. Must match the input tensor's feature dimension. This determines the size of the positional encoding vectors.

- ``max_len`` (int): Maximum sequence length to pre-compute encodings for. The positional encodings are pre-computed up to this length for efficiency. If your sequence is longer, the encoding will be truncated.

**Input/Output:**

- **Input**: Tensor with shape ``(batch_size, sequence_length, d_model)``. The feature dimension must match ``d_model``.

- **Output**: Tensor with the same shape ``(batch_size, sequence_length, d_model)``. The positional encoding is added (element-wise) to the input embeddings.

**When to Use:**

- Building transformer-based models (e.g., SingleShotTransformer)
- Need to encode temporal position information
- Working with sequences where order matters

**How It Works:**

Uses sinusoidal functions (sine and cosine) with different frequencies to encode position. Each dimension of the encoding corresponds to a different frequency, allowing the model to learn relative positions. The encoding is pre-computed during initialization and stored as a buffer.

**Architecture:**

Based on the original Transformer paper (Vaswani et al., 2017), modified for batch-first format and without dropout. The encoding uses:

- Sine functions for even dimensions: ``sin(pos / 10000^(2i/d_model))``
- Cosine functions for odd dimensions: ``cos(pos / 10000^(2i/d_model))``

where ``pos`` is the position and ``i`` is the dimension index.

**Example:**

.. code-block:: python

   import torch
   from deep_time_series.layer import PositionalEncoding

   layer = PositionalEncoding(d_model=128, max_len=100)
   x = torch.randn(32, 50, 128)  # (batch, time, d_model)
   y = layer(x)  # (32, 50, 128) - positional encoding added

**Key Properties:**

- **Sinusoidal**: Uses sin/cos functions for smooth position encoding
- **Fixed**: The encoding is deterministic and not learnable (stored as buffer)
- **Relative Positions**: The encoding allows the model to understand relative distances between time steps
- **Additive**: The encoding is added (not concatenated) to the input embeddings

**Note:**

The encoding is added (not concatenated) to the input embeddings. This allows the model to learn how to combine positional and feature information. The positional encodings are registered as buffers, so they are automatically moved to the correct device when the model is moved.

.. autoclass:: deep_time_series.layer.PositionalEncoding
   :members:
   :undoc-members:
   :show-inheritance:
