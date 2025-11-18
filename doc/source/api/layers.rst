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

**When to Use:**

- Implementing causal convolutions (e.g., in DilatedCNN)
- Ensuring temporal order is preserved
- Preventing information leakage from future to past

**How It Works:**

Adds zeros to the left (beginning) of the sequence, shifting the original data to the right. This allows convolutions to process the sequence while maintaining causality.

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

**When to Use:**

- Building transformer-based models (e.g., SingleShotTransformer)
- Need to encode temporal position information
- Working with sequences where order matters

**How It Works:**

Uses sinusoidal functions (sine and cosine) with different frequencies to encode position. Each dimension of the encoding corresponds to a different frequency, allowing the model to learn relative positions.

**Architecture:**

Based on the original Transformer paper, modified for batch-first format and without dropout.

**Example:**

.. code-block:: python

   import torch
   from deep_time_series.layer import PositionalEncoding

   layer = PositionalEncoding(d_model=128, max_len=100)
   x = torch.randn(32, 50, 128)  # (batch, time, d_model)
   y = layer(x)  # (32, 50, 128) - positional encoding added

**Key Properties:**

- **Sinusoidal**: Uses sin/cos functions for smooth position encoding
- **Learnable**: While the encoding is fixed, the model can learn to use it
- **Relative Positions**: The encoding allows the model to understand relative distances

**Parameters:**

- ``d_model``: Embedding dimension (must match input dimension)
- ``max_len``: Maximum sequence length to pre-compute encodings for

**Note:**

The encoding is added (not concatenated) to the input embeddings. This allows the model to learn how to combine positional and feature information.

.. autoclass:: deep_time_series.layer.PositionalEncoding
   :members:
   :undoc-members:
   :show-inheritance:
