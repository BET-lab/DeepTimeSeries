API Reference
=============

This page provides detailed documentation for all public classes, functions, and modules in DeepTimeSeries.

The API is organized into the following sections:

Core Modules
------------
Base classes for building forecasting models. These include :class:`~deep_time_series.core.ForecastingModule` (the foundation for all models), head classes for producing predictions, and metric tracking utilities.

Chunk Specification
--------------------
Classes for defining and extracting time windows from time series data. Chunks specify what parts of the data are used for encoding (input), decoding (during prediction), and labels (targets).

Dataset
-------
The :class:`~deep_time_series.dataset.TimeSeriesDataset` class provides a PyTorch-compatible dataset for loading time series data with chunk-based extraction.

Data Transformation
--------------------
Preprocessing utilities, primarily :class:`~deep_time_series.transform.ColumnTransformer` for applying sklearn-style transformers to specific columns of DataFrames.

Models
------
Pre-implemented forecasting models including MLP, Dilated CNN, RNN variants (LSTM, GRU), and Transformer. All models support both deterministic and probabilistic forecasting.

Layers
------
Custom neural network layers used by the forecasting models, such as positional encoding for transformers and padding layers for causal convolutions.

Utilities
--------
Helper functions for data manipulation, dictionary operations, and visualization utilities for time series data.

Typical Usage Flow
------------------

1. **Preprocess data** using :class:`~deep_time_series.transform.ColumnTransformer`
2. **Create a model** (e.g., :class:`~deep_time_series.model.MLP`) which automatically generates chunk specifications
3. **Create a dataset** using :class:`~deep_time_series.dataset.TimeSeriesDataset` with the model's chunk specifications
4. **Train the model** using PyTorch Lightning's Trainer
5. **Use** :class:`~deep_time_series.chunk.ChunkInverter` to convert model outputs back to DataFrames

.. toctree::
   :maxdepth: 2

   core.rst
   chunk.rst
   dataset.rst
   transform.rst
   models.rst
   layers.rst
   utilities.rst
