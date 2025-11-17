DeepTimeSeries
============================================

Deep learning library for time series forecasting based on PyTorch and PyTorch Lightning.

Overview
--------

DeepTimeSeries is a comprehensive library designed for time series forecasting using deep learning models.
It provides a logical framework for designing and implementing various deep learning architectures
specifically tailored for time series data.

The library is built on PyTorch and PyTorch Lightning, offering both high-level APIs for beginners
and flexible low-level components for intermediate to advanced users who need to customize their forecasting models.

Key Features
------------

- **Modular Architecture**: Clean separation between encoding, decoding, and prediction components
- **Multiple Model Support**: Pre-implemented models including MLP, Dilated CNN, RNN variants (LSTM, GRU), and Transformer
- **Flexible Data Handling**: Pandas DataFrame-based data processing with chunk-based extraction
- **Data Preprocessing**: Built-in ``ColumnTransformer`` for feature scaling and transformation
- **Probabilistic Forecasting**: Support for both deterministic and probabilistic predictions
- **PyTorch Lightning Integration**: Seamless integration with Lightning for training, validation, and testing

Why DeepTimeSeries?
-------------------

DeepTimeSeries is inspired by libraries such as ``Darts`` and ``PyTorch Forecasting``.
So why was DeepTimeSeries developed?

The design philosophy of DeepTimeSeries is as follows:

**We present logical guidelines for designing various deep learning models for time series forecasting**

Our main target users are intermediate-level users who need to develop deep learning models for time series forecasting.
We provide solutions to many problems that deep learning modelers face because of the uniqueness of time series data.

We additionally implement a high-level API, which allows comparatively beginners to use models that have already been implemented.

Supported Models
---------------

The library provides several pre-implemented models:

- **MLP**: Multi-layer perceptron for time series forecasting
- **Dilated CNN**: Dilated convolutional neural network for capturing long-range dependencies
- **RNN**: Vanilla RNN, LSTM, and GRU variants for sequential modeling
- **Transformer**: Single-shot transformer architecture for time series forecasting

All models support both deterministic and probabilistic forecasting, and can handle both target and non-target features.

.. toctree::
   :hidden:

   User Guide <user_guide/index>
   Tutorial <tutorials/index>
   API Reference <_autosummary/deep_time_series>
