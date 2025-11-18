Models
======

The model module provides pre-implemented forecasting models. All models inherit from :class:`~deep_time_series.core.ForecastingModule` and support both deterministic and probabilistic forecasting.

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
---

Multi-layer perceptron model for time series forecasting. This model flattens the encoding window and processes it through fully connected layers.

.. autoclass:: deep_time_series.model.MLP
   :members:
   :undoc-members:
   :show-inheritance:

Dilated CNN
-----------

Dilated convolutional neural network for capturing long-range dependencies in time series. Uses dilated convolutions with exponentially increasing dilation rates to capture patterns at multiple time scales.

.. autoclass:: deep_time_series.model.DilatedCNN
   :members:
   :undoc-members:
   :show-inheritance:

RNN
---

Recurrent neural network model supporting vanilla RNN, LSTM, and GRU variants. Uses an encoder-decoder architecture where the encoder processes the encoding window and the decoder generates predictions autoregressively.

.. autoclass:: deep_time_series.model.RNN
   :members:
   :undoc-members:
   :show-inheritance:

Single Shot Transformer
-----------------------

Transformer-based model for time series forecasting using encoder-decoder architecture. The encoder processes the encoding window, and the decoder generates all predictions in a single forward pass (single-shot) rather than autoregressively.

.. autoclass:: deep_time_series.model.SingleShotTransformer
   :members:
   :undoc-members:
   :show-inheritance:

