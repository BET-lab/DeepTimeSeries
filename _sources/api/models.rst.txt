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

MLP
---

Multi-layer perceptron model for time series forecasting. This model flattens the encoding window and processes it through fully connected layers.

**Purpose:**

A simple feedforward neural network that treats time series forecasting as a sequence-to-sequence problem. Good for baseline comparisons and simple patterns.

**Architecture:**

1. Flattens the encoding window into a 1D vector
2. Passes through multiple fully connected layers with activation
3. Uses autoregressive decoding: predicts one step at a time, feeding predictions back

**Initialization Parameters:**

**Required Parameters:**

- ``hidden_size`` (int): Number of neurons in hidden layers. Typically 32-256, larger for more complex patterns.

- ``encoding_length`` (int): Length of the input window. Typically 10-50 time steps for MLP.

- ``decoding_length`` (int): Length of the prediction horizon.

- ``target_names`` (list[str]): List of column names to predict. These are the target variables.

- ``nontarget_names`` (list[str]): List of column names for exogenous variables. Can be empty list if no exogenous variables.

- ``n_hidden_layers`` (int): Number of hidden layers. Typically 1-4 layers. Deeper networks may overfit.

**Optional Parameters:**

- ``activation`` (nn.Module): Activation function class. Default is ``nn.ELU``. Common choices: ``nn.ReLU``, ``nn.ELU``, ``nn.GELU``.

- ``dropout_rate`` (float): Dropout probability. Default is 0.0. Use 0.0-0.3 for regularization.

- ``lr`` (float): Learning rate. Default is 1e-3.

- ``optimizer`` (torch.optim.Optimizer): Optimizer class. Default is ``torch.optim.Adam``.

- ``optimizer_options`` (dict | None): Additional optimizer options. Default is ``None`` (empty dict).

- ``loss_fn`` (Callable | None): Loss function. Default is ``nn.MSELoss()``.

- ``metrics`` (Metric | list[Metric] | dict[str, Metric] | None): Metrics to track. Default is ``None``.

- ``head`` (BaseHead | None): Custom head. If provided, overrides default head creation. Default is ``None``.

**Methods:**

- **``encode(inputs)``**: Processes the encoding window. Concatenates target and non-target features, then flattens them. Returns a dictionary with key ``'x'`` containing the flattened features.

- **``decode(inputs)``**: Autoregressive decoding. For each time step:
  1. Processes the current window through the MLP body
  2. Generates prediction using the head
  3. Updates the window by shifting and appending the prediction
  4. Returns accumulated predictions from the head

- **``make_chunk_specs()``**: Generates chunk specifications for this model. Creates encoding and label chunks for targets, and encoding/decoding chunks for non-targets if present.

- **``configure_optimizers()``**: PyTorch Lightning method that configures the optimizer. Returns the optimizer instance.

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

**Purpose:**

A convolutional model that efficiently captures long-range dependencies through dilated convolutions. Each layer operates at a different time scale, allowing the model to capture both local and global patterns.

**Architecture:**

1. Stacked dilated 1D convolutions with exponentially increasing dilation rates
2. Each layer captures patterns at different time scales
3. Left padding ensures causal (non-leaking) convolutions
4. Autoregressive decoding for predictions

**Initialization Parameters:**

**Required Parameters:**

- ``hidden_size`` (int): Number of channels in convolutional layers. Typically 32-256.

- ``encoding_length`` (int): Length of the input window. Can handle 50-500+ time steps effectively.

- ``decoding_length`` (int): Length of the prediction horizon.

- ``target_names`` (list[str]): List of column names to predict.

- ``nontarget_names`` (list[str]): List of column names for exogenous variables.

- ``dilation_base`` (int): Base for exponential dilation. Typically 2 or 3. Controls how fast dilation increases across layers. Must satisfy ``dilation_base <= kernel_size``.

- ``kernel_size`` (int): Size of convolutional kernel. Typically 2-5. Larger kernels capture broader patterns. Must satisfy ``kernel_size >= dilation_base``.

**Optional Parameters:**

- ``activation`` (nn.Module): Activation function class. Default is ``nn.ELU``.

- ``dropout_rate`` (float): Dropout probability. Default is 0.0.

- ``lr`` (float): Learning rate. Default is 1e-3.

- ``optimizer`` (torch.optim.Optimizer): Optimizer class. Default is ``torch.optim.Adam``.

- ``optimizer_options`` (dict | None): Additional optimizer options. Default is ``None``.

- ``loss_fn`` (Callable | None): Loss function. Default is ``nn.MSELoss()``.

- ``metrics`` (Metric | list[Metric] | dict[str, Metric] | None): Metrics to track. Default is ``None``.

- ``head`` (BaseHead | None): Custom head. Default is ``None``.

**Methods:**

- **``encode(inputs)``**: Processes the encoding window through stacked dilated convolutions. Concatenates target and non-target features, applies convolutions with increasing dilation rates. Returns a dictionary with processed features.

- **``decode(inputs)``**: Autoregressive decoding similar to MLP, but uses the convolutional features. Generates predictions step by step.

- **``make_chunk_specs()``**: Generates chunk specifications. Similar to MLP but with different time ranges for non-targets.

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

**Important Constraint:**

``kernel_size >= dilation_base`` must be satisfied. The number of layers is automatically calculated based on encoding_length, dilation_base, and kernel_size.

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

**Purpose:**

A recurrent model that processes sequences step by step, naturally modeling temporal dependencies. Supports multiple RNN variants for different use cases.

**Architecture:**

1. Encoder RNN processes the encoding window sequentially
2. Final hidden state captures the context
3. Decoder RNN generates predictions autoregressively
4. Same RNN instance used for both encoding and decoding

**Initialization Parameters:**

**Required Parameters:**

- ``hidden_size`` (int): Size of hidden state. Typically 32-256, larger for more complex patterns.

- ``encoding_length`` (int): Length of the input window.

- ``decoding_length`` (int): Length of the prediction horizon.

- ``target_names`` (list[str]): List of column names to predict.

- ``nontarget_names`` (list[str]): List of column names for exogenous variables.

- ``n_layers`` (int): Number of RNN layers. Typically 1-3 layers. Deeper may help but increases complexity.

- ``rnn_class`` (nn.Module): RNN class to use. Must be ``nn.RNN``, ``nn.LSTM``, or ``nn.GRU``. Recommended: ``nn.LSTM`` or ``nn.GRU``.

**Optional Parameters:**

- ``dropout_rate`` (float): Dropout probability applied between RNN layers. Default is 0.0. Use 0.0-0.3.

- ``lr`` (float): Learning rate. Default is 1e-3.

- ``optimizer`` (torch.optim.Optimizer): Optimizer class. Default is ``torch.optim.Adam``.

- ``optimizer_options`` (dict | None): Additional optimizer options. Default is ``None``.

- ``loss_fn`` (Callable | None): Loss function. Default is ``nn.MSELoss()``.

- ``metrics`` (Metric | list[Metric] | dict[str, Metric] | None): Metrics to track. Default is ``None``.

- ``head`` (BaseHead | None): Custom head. Default is ``None``.

**Methods:**

- **``encode(inputs)``**: Processes the encoding window through the RNN encoder. Concatenates target and non-target features, processes sequentially, and returns the final hidden state and memory (for LSTM/GRU).

- **``decode(inputs)``**: Autoregressive decoding using the RNN decoder. Starts from the encoder's final hidden state, generates predictions step by step, and updates the hidden state.

- **``make_chunk_specs()``**: Generates chunk specifications for this model.

**RNN Variants:**

- **Vanilla RNN** (``nn.RNN``): Simple but may suffer from vanishing gradients. Not recommended for long sequences.

- **LSTM** (``nn.LSTM``): Long short-term memory, handles long dependencies well. Recommended for most cases.

- **GRU** (``nn.GRU``): Gated recurrent unit, similar to LSTM but simpler and faster. Good alternative to LSTM.

**When to Use:**

- Sequential patterns with clear temporal dependencies
- Need to model order and sequence structure explicitly
- Variable-length sequences (though fixed length in current implementation)
- Natural fit for time series data

**Strengths:**

- Natural modeling of sequential dependencies
- LSTM/GRU handle long-range dependencies well
- Interpretable hidden states
- Well-established architecture

**Limitations:**

- Sequential processing (slower than parallel architectures)
- May struggle with very long sequences
- Requires careful initialization and regularization

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

**Purpose:**

A transformer-based model that uses attention mechanisms to capture complex temporal relationships. Unlike other models, it generates all predictions simultaneously rather than autoregressively, making inference faster.

**Architecture:**

1. Encoder: Multi-head self-attention processes the encoding window
2. Decoder: Multi-head cross-attention between decoder inputs and encoder outputs
3. Positional encoding adds temporal information
4. Single-shot prediction: All future steps predicted simultaneously

**Initialization Parameters:**

**Required Parameters:**

- ``encoding_length`` (int): Length of the input window.

- ``decoding_length`` (int): Length of the prediction horizon.

- ``target_names`` (list[str]): List of column names to predict.

- ``nontarget_names`` (list[str]): List of column names for exogenous variables.

- ``d_model`` (int): Embedding dimension. Typically 64-512. Larger values provide more capacity but require more computation. Must be divisible by ``n_heads``.

- ``n_heads`` (int): Number of attention heads. Typically 4-16. ``d_model`` must be divisible by ``n_heads``.

- ``n_layers`` (int): Number of transformer layers. Typically 2-6. Deeper for more complex patterns.

**Optional Parameters:**

- ``dim_feedforward`` (int | None): Dimension of feedforward network. Default is ``4 * d_model``. Typically 4 times d_model.

- ``dropout_rate`` (float): Dropout probability. Default is 0.0. Use 0.1-0.3 for regularization.

- ``lr`` (float): Learning rate. Default is 1e-3.

- ``optimizer`` (torch.optim.Optimizer): Optimizer class. Default is ``torch.optim.Adam``.

- ``optimizer_options`` (dict | None): Additional optimizer options. Default is ``None``.

- ``loss_fn`` (Callable | None): Loss function. Default is ``nn.MSELoss()``.

- ``metrics`` (Metric | list[Metric] | dict[str, Metric] | None): Metrics to track. Default is ``None``.

- ``head`` (BaseHead | None): Custom head. Default is ``None``.

**Methods:**

- **``encode(inputs)``**: Processes the encoding window through the transformer encoder. Applies positional encoding, then processes through multiple encoder layers with self-attention. Returns encoder outputs.

- **``decode(inputs)``**: Processes decoder inputs through the transformer decoder. Uses cross-attention to attend to encoder outputs. Generates all predictions in a single forward pass (not autoregressive).

- **``make_chunk_specs()``**: Generates chunk specifications for this model.

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
- Memory usage scales with sequence length squared (O(n²) attention complexity)

**Important Notes:**

- Unlike other models, this generates all predictions in one forward pass rather than autoregressively. This makes inference faster but may reduce accuracy for very long prediction horizons.

- The model uses ``nn.TransformerEncoder`` and ``nn.TransformerDecoder`` from PyTorch, with batch-first format.

- Positional encoding is applied to both encoder and decoder inputs to provide temporal information.

.. autoclass:: deep_time_series.model.SingleShotTransformer
   :members:
   :undoc-members:
   :show-inheritance:
