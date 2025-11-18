API Reference
=============

This page provides detailed documentation for all public classes, functions, and modules in DeepTimeSeries.

Core Modules
------------

The core module provides the base classes for building forecasting models.

.. automodule:: deep_time_series.core
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.core.ForecastingModule
   deep_time_series.core.BaseHead
   deep_time_series.core.Head
   deep_time_series.core.DistributionHead
   deep_time_series.core.MetricModule

Chunk Specification
-------------------

The chunk module provides classes for defining and extracting chunks from time series data.

.. automodule:: deep_time_series.chunk
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.chunk.BaseChunkSpec
   deep_time_series.chunk.EncodingChunkSpec
   deep_time_series.chunk.DecodingChunkSpec
   deep_time_series.chunk.LabelChunkSpec
   deep_time_series.chunk.ChunkExtractor
   deep_time_series.chunk.ChunkInverter

Dataset
-------

The dataset module provides the TimeSeriesDataset class for loading and processing time series data.

.. automodule:: deep_time_series.dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.dataset.TimeSeriesDataset

Data Transformation
-------------------

The transform module provides data preprocessing utilities.

.. automodule:: deep_time_series.transform
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.transform.ColumnTransformer

Models
------

The model module provides pre-implemented forecasting models.

.. automodule:: deep_time_series.model
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.model.MLP
   deep_time_series.model.DilatedCNN
   deep_time_series.model.RNN
   deep_time_series.model.SingleShotTransformer

MLP
~~~

Multi-layer perceptron model for time series forecasting.

.. autoclass:: deep_time_series.model.MLP
   :members:
   :undoc-members:
   :show-inheritance:

Dilated CNN
~~~~~~~~~~~

Dilated convolutional neural network for capturing long-range dependencies in time series.

.. autoclass:: deep_time_series.model.DilatedCNN
   :members:
   :undoc-members:
   :show-inheritance:

RNN
~~~

Recurrent neural network model supporting vanilla RNN, LSTM, and GRU variants.

.. autoclass:: deep_time_series.model.RNN
   :members:
   :undoc-members:
   :show-inheritance:

Single Shot Transformer
~~~~~~~~~~~~~~~~~~~~~~~

Transformer-based model for time series forecasting using encoder-decoder architecture.

.. autoclass:: deep_time_series.model.SingleShotTransformer
   :members:
   :undoc-members:
   :show-inheritance:

Layers
------

The layer module provides custom neural network layers for time series models.

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

Utilities
---------

Utility Functions
~~~~~~~~~~~~~~~~~~

The util module provides helper functions for data manipulation and dictionary operations.

.. automodule:: deep_time_series.util
   :members:
   :undoc-members:

.. autosummary::
   :toctree: _autosummary
   :template: function.rst
   :recursive:

   deep_time_series.util.logical_and_for_set_list
   deep_time_series.util.logical_or_for_set_list
   deep_time_series.util.merge_dicts
   deep_time_series.util.merge_data_frames

Plotting
~~~~~~~~

The plotting module provides visualization utilities for time series data.

.. automodule:: deep_time_series.plotting
   :members:
   :undoc-members:

.. autosummary::
   :toctree: _autosummary
   :template: function.rst
   :recursive:

   deep_time_series.plotting.plot_chunks
