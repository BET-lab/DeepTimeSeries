Models
======

The model module provides pre-implemented forecasting models. All models inherit from :class:`~deep_time_series.core.ForecastingModule` and support both deterministic and probabilistic forecasting.

**Common Features:**

All models share these capabilities:
- **Deterministic Forecasting**: Point predictions using :class:`~deep_time_series.core.Head`
- **Probabilistic Forecasting**: Distribution-based predictions using :class:`~deep_time_series.core.DistributionHead`
- **Target and Non-target Features**: Support for both features to predict and exogenous variables
- **PyTorch Lightning Integration**: Automatic training, validation, and testing loops
- **Customizable**: All hyperparameters can be customized, including optimizer, loss function, and metrics

**Model Selection Guide:**

- **MLP**: Simple and fast, good for short-term dependencies
- **Dilated CNN**: Captures long-range dependencies efficiently, good for long sequences
- **RNN**: Natural for sequential data, supports LSTM/GRU variants
- **Single Shot Transformer**: Best for complex patterns, generates all predictions at once

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

**Architecture:**

1. Flattens the encoding window into a 1D vector
2. Passes through multiple fully connected layers with activation
3. Uses autoregressive decoding: predicts one step at a time, feeding predictions back

**When to Use:**

- Short-term dependencies (encoding_length < 50)
- Simple patterns that don't require complex temporal modeling
- Fast training and inference required
- Baseline model for comparison

**Strengths:**

- Simple and interpretable
- Fast training and inference
- Low memory requirements
- Works well with small datasets

**Limitations:**

- Limited ability to capture long-range dependencies
- Doesn't explicitly model temporal structure
- May struggle with complex patterns

**Parameter Guidelines:**

- ``hidden_size``: Typically 32-256, larger for more complex patterns
- ``n_hidden_layers``: 1-4 layers, deeper networks may overfit
- ``encoding_length``: 10-50 time steps, shorter for simple patterns
- ``dropout_rate``: 0.0-0.3, use dropout for regularization

**Example:**

.. code-block:: python

   from deep_time_series.model import MLP
   import torch.nn as nn
   from torchmetrics import MeanAbsoluteError

   model = MLP(
       hidden_size=64,
       encoding_length=20,
       decoding_length=10,
       target_names=['temperature'],
       nontarget_names=['humidity', 'pressure'],
       n_hidden_layers=2,
       activation=nn.ReLU,
       dropout_rate=0.1,
       lr=1e-3,
       loss_fn=nn.MSELoss(),
       metrics=MeanAbsoluteError(),
   )

.. autoclass:: deep_time_series.model.MLP
   :members:
   :undoc-members:
   :show-inheritance:

Dilated CNN
-----------

Dilated convolutional neural network for capturing long-range dependencies in time series. Uses dilated convolutions with exponentially increasing dilation rates to capture patterns at multiple time scales.

**Architecture:**

1. Stacked dilated 1D convolutions with exponentially increasing dilation rates
2. Each layer captures patterns at different time scales
3. Left padding ensures causal (non-leaking) convolutions
4. Autoregressive decoding for predictions

**When to Use:**

- Long-range dependencies (encoding_length > 50)
- Multiple time scales in the data
- Need efficient processing of long sequences
- Want to capture both local and global patterns

**Strengths:**

- Efficiently captures long-range dependencies
- Parallel processing during encoding
- Good performance on long sequences
- Less prone to vanishing gradients than RNNs

**Limitations:**

- Requires careful tuning of dilation_base and kernel_size
- May miss very short-term patterns if dilation is too large
- Less interpretable than simpler models

**Parameter Guidelines:**

- ``dilation_base``: Typically 2 or 3, controls how fast dilation increases
- ``kernel_size``: 2-5, larger for capturing broader patterns
- ``hidden_size``: 32-256, similar to MLP
- ``encoding_length``: Can handle 50-500+ time steps effectively

**Important Constraint:**

``kernel_size >= dilation_base`` must be satisfied.

**Example:**

.. code-block:: python

   from deep_time_series.model import DilatedCNN
   import torch.nn as nn

   model = DilatedCNN(
       hidden_size=128,
       encoding_length=100,
       decoding_length=20,
       target_names=['temperature'],
       nontarget_names=['humidity'],
       dilation_base=2,
       kernel_size=3,
       activation=nn.ELU,
       dropout_rate=0.1,
   )

.. autoclass:: deep_time_series.model.DilatedCNN
   :members:
   :undoc-members:
   :show-inheritance:

RNN
---

Recurrent neural network model supporting vanilla RNN, LSTM, and GRU variants. Uses an encoder-decoder architecture where the encoder processes the encoding window and the decoder generates predictions autoregressively.

**Architecture:**

1. Encoder RNN processes the encoding window sequentially
2. Final hidden state captures the context
3. Decoder RNN generates predictions autoregressively
4. Same RNN instance used for both encoding and decoding

**When to Use:**

- Sequential patterns with clear temporal dependencies
- Need to model order and sequence structure explicitly
- Variable-length sequences (though fixed length in current implementation)
- Natural fit for time series data

**RNN Variants:**

- **Vanilla RNN** (``nn.RNN``): Simple but may suffer from vanishing gradients
- **LSTM** (``nn.LSTM``): Long short-term memory, handles long dependencies well
- **GRU** (``nn.GRU``): Gated recurrent unit, similar to LSTM but simpler

**Strengths:**

- Natural modeling of sequential dependencies
- LSTM/GRU handle long-range dependencies well
- Interpretable hidden states
- Well-established architecture

**Limitations:**

- Sequential processing (slower than parallel architectures)
- May struggle with very long sequences
- Requires careful initialization and regularization

**Parameter Guidelines:**

- ``rnn_class``: Use ``nn.LSTM`` or ``nn.GRU`` for better performance than ``nn.RNN``
- ``n_layers``: 1-3 layers, deeper may help but increases complexity
- ``hidden_size``: 32-256, larger for more complex patterns
- ``dropout_rate``: 0.0-0.3, applied between RNN layers

**Example:**

.. code-block:: python

   from deep_time_series.model import RNN
   import torch.nn as nn

   # LSTM variant
   model = RNN(
       hidden_size=128,
       encoding_length=50,
       decoding_length=10,
       target_names=['temperature'],
       nontarget_names=['humidity'],
       n_layers=2,
       rnn_class=nn.LSTM,  # or nn.GRU, nn.RNN
       dropout_rate=0.2,
   )

.. autoclass:: deep_time_series.model.RNN
   :members:
   :undoc-members:
   :show-inheritance:

Single Shot Transformer
-----------------------

Transformer-based model for time series forecasting using encoder-decoder architecture. The encoder processes the encoding window, and the decoder generates all predictions in a single forward pass (single-shot) rather than autoregressively.

**Architecture:**

1. Encoder: Multi-head self-attention processes the encoding window
2. Decoder: Multi-head cross-attention between decoder inputs and encoder outputs
3. Positional encoding adds temporal information
4. Single-shot prediction: All future steps predicted simultaneously

**When to Use:**

- Complex patterns requiring attention mechanisms
- Need to model relationships between distant time steps
- Want parallel prediction (faster inference than autoregressive)
- Large datasets with sufficient computational resources

**Strengths:**

- Captures complex temporal relationships via attention
- Parallel prediction (faster inference)
- State-of-the-art performance potential
- Flexible architecture

**Limitations:**

- Requires more data and computation
- May overfit on small datasets
- Less interpretable than simpler models
- Memory usage scales with sequence length squared

**Parameter Guidelines:**

- ``d_model``: 64-512, embedding dimension (larger = more capacity)
- ``n_heads``: 4-16, number of attention heads (d_model must be divisible by n_heads)
- ``n_layers``: 2-6, deeper for more complex patterns
- ``dim_feedforward``: Typically 4 * d_model (default)
- ``dropout_rate``: 0.1-0.3, important for regularization

**Example:**

.. code-block:: python

   from deep_time_series.model import SingleShotTransformer
   import torch.nn as nn

   model = SingleShotTransformer(
       encoding_length=100,
       decoding_length=20,
       target_names=['temperature'],
       nontarget_names=['humidity', 'pressure'],
       d_model=128,
       n_heads=8,
       n_layers=4,
       dim_feedforward=512,  # 4 * d_model
       dropout_rate=0.1,
   )

**Note:**

Unlike other models, this generates all predictions in one forward pass rather than autoregressively. This makes inference faster but may reduce accuracy for very long prediction horizons.

.. autoclass:: deep_time_series.model.SingleShotTransformer
   :members:
   :undoc-members:
   :show-inheritance:
