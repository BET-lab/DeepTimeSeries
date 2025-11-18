Layers
======

The layer module provides custom neural network layers for time series models. These layers are utility components used by the forecasting models.

.. automodule:: deep_time_series.layer
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.layer.Unsqueesze
   deep_time_series.layer.Permute
   deep_time_series.layer.LeftPadding1D
   deep_time_series.layer.PositionalEncoding

Unsqueesze
~~~~~~~~~~

A layer that adds a dimension to the input tensor at the specified position.

.. autoclass:: deep_time_series.layer.Unsqueesze
   :members:
   :undoc-members:
   :show-inheritance:

Permute
~~~~~~~

A layer that permutes the dimensions of the input tensor according to the specified order.

.. autoclass:: deep_time_series.layer.Permute
   :members:
   :undoc-members:
   :show-inheritance:

LeftPadding1D
~~~~~~~~~~~~~

A layer that adds left padding to 1D sequences. Useful for causal convolutions in time series models.

.. autoclass:: deep_time_series.layer.LeftPadding1D
   :members:
   :undoc-members:
   :show-inheritance:

PositionalEncoding
~~~~~~~~~~~~~~~~~~

Positional encoding layer for transformer models. Adds positional information to input embeddings.

.. autoclass:: deep_time_series.layer.PositionalEncoding
   :members:
   :undoc-members:
   :show-inheritance:

