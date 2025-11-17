Design of DeepTimeSeries
=========================

This page describes the design of DeepTimeSeries and the concepts for understanding the design.

1. Encoding and Decoding
------------------------

DeepTimeSeries uses an encoder-decoder architecture for time series forecasting.
This design separates the input processing (encoding) from the output generation (decoding),
providing flexibility and clarity in model design.

Encoding Phase
~~~~~~~~~~~~~~

The encoding phase processes historical time series data to extract meaningful representations.
The encoder receives input data from the **encoding window** (defined by ``encoding_length``)
and produces a context representation that captures the patterns and dependencies in the historical data.

The encoding process:

- Takes input from ``encoding.targets`` and optionally ``encoding.nontargets``
- Processes the data through the encoder network (e.g., RNN, CNN, Transformer encoder)
- Produces a context representation (e.g., hidden states, memory, encoded features)

Decoding Phase
~~~~~~~~~~~~~~

The decoding phase generates future predictions based on the encoded context.
The decoder uses the context from the encoder and optionally future non-target features
to generate predictions for the **decoding window** (defined by ``decoding_length``).

The decoding process:

- Receives the encoded context and optionally ``decoding.nontargets``
- Generates predictions step-by-step for each time step in the decoding window
- Can use teacher forcing during training (using ground truth) or autoregressive generation during inference

Key Design Principles
~~~~~~~~~~~~~~~~~~~~~

- **Separation of Concerns**: Encoding and decoding are clearly separated, making it easy to understand and modify each component
- **Flexible Input Handling**: Models can handle both target variables and non-target features
- **Autoregressive Generation**: Decoders generate predictions step-by-step, allowing for conditional generation

2. Chunk Specification System
------------------------------

The chunk specification system is a core concept in DeepTimeSeries that defines
how time series data is extracted and organized for model training and inference.

Chunk Types
~~~~~~~~~~~

There are three types of chunk specifications:

**EncodingChunkSpec**
    Defines the input window for the encoder. This specifies which time steps
    and features are used as input to the encoder.

**DecodingChunkSpec**
    Defines the input window for the decoder. This typically includes future
    non-target features that are known at prediction time.

**LabelChunkSpec**
    Defines the target window for prediction. This specifies which time steps
    and features should be predicted.

Chunk Specification Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each chunk specification has the following properties:

- **tag**: A unique identifier for the chunk (e.g., ``'targets'``, ``'nontargets'``)
- **names**: List of feature names (columns) to extract
- **range_**: Tuple ``(start, end)`` defining the time window relative to a reference point
- **dtype**: Data type for the extracted data (typically ``np.float32``)

Example
~~~~~~~

For a model with ``encoding_length=10`` and ``decoding_length=5``:

.. code-block:: python

    EncodingChunkSpec(
        tag='targets',
        names=['target'],
        range_=(0, 10),  # Time steps 0-9
        dtype=np.float32
    )
    
    LabelChunkSpec(
        tag='targets',
        names=['target'],
        range_=(10, 15),  # Time steps 10-14
        dtype=np.float32
    )

ChunkExtractor and ChunkInverter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **ChunkExtractor**: Extracts chunks from DataFrames based on chunk specifications
- **ChunkInverter**: Converts model outputs (tensors) back to DataFrame format

3. ForecastingModule Architecture
---------------------------------

``ForecastingModule`` is the base class for all forecasting models in DeepTimeSeries.
It extends PyTorch Lightning's ``LightningModule``, providing automatic training,
validation, and testing loops.

Core Components
~~~~~~~~~~~~~~~

**Encoding and Decoding Methods**
    - ``encode()``: Must be implemented by subclasses to process input data
    - ``decode_train()``: Decoding logic for training (defaults to ``decode_eval()``)
    - ``decode_eval()``: Decoding logic for evaluation/inference
    - ``decode()``: Automatically selects training or evaluation decoding based on mode

**Head Management**
    - ``heads``: List of ``BaseHead`` instances for multi-head models
    - ``head``: Convenience property for single-head models

**Chunk Specification**
    - ``make_chunk_specs()``: Generates chunk specifications based on model parameters
    - ``encoding_length``: Length of the encoding window
    - ``decoding_length``: Length of the decoding window

Automatic Training Loop
~~~~~~~~~~~~~~~~~~~~~~~

The ``ForecastingModule`` automatically handles:

- **Loss Calculation**: Aggregates losses from all heads with their respective weights
- **Metric Tracking**: Updates and computes metrics for each head
- **Logging**: Logs losses and metrics to PyTorch Lightning's logger

Training Flow
~~~~~~~~~~~~~

1. Input batch is passed to ``forward()``
2. ``encode()`` processes the encoding window
3. ``decode()`` generates predictions using the encoded context
4. ``calculate_loss()`` computes the total loss from all heads
5. Metrics are updated and logged automatically

4. Head System
--------------

The head system provides a flexible way to define output layers and loss functions
for forecasting models. It supports both deterministic and probabilistic forecasting.

BaseHead
~~~~~~~~

``BaseHead`` is the abstract base class for all heads. It defines the interface
that all heads must implement:

- **tag**: Unique identifier for the head (automatically prefixed with ``'head.'``)
- **loss_weight**: Weight for the head's loss in the total loss calculation
- **metrics**: Optional metrics to track (e.g., MAE, MSE, RMSE)

Key Methods
~~~~~~~~~~~

- ``forward()``: Processes input and generates output
- ``get_outputs()``: Returns accumulated outputs as a dictionary
- ``reset()``: Resets internal state (e.g., accumulated outputs)
- ``calculate_loss()``: Computes loss given outputs and batch

Head Types
~~~~~~~~~~

**Head (Deterministic)**
    Standard head for deterministic forecasting:
    
    - Takes an ``output_module`` (e.g., ``nn.Linear``) and a ``loss_fn``
    - Generates point predictions
    - Supports custom loss functions and metrics
    
**DistributionHead (Probabilistic)**
    Head for probabilistic forecasting:
    
    - Takes a PyTorch distribution class (e.g., ``torch.distributions.Normal``)
    - Automatically creates linear layers for distribution parameters
    - Uses negative log-likelihood as the loss function
    - Supports sampling from the distribution

Multi-Head Models
~~~~~~~~~~~~~~~~~

Models can have multiple heads for different prediction tasks:

- Each head has its own tag, loss weight, and metrics
- Total loss is the weighted sum of all head losses
- Metrics are tracked separately for each head

5. Data Flow
------------

Understanding the complete data flow helps in designing and debugging models.

Data Preparation
~~~~~~~~~~~~~~~

1. **Load Data**: Load time series data as pandas DataFrame(s)
2. **Preprocess**: Apply ``ColumnTransformer`` for scaling/transformation
3. **Create Dataset**: Use ``TimeSeriesDataset`` with chunk specifications
4. **Create DataLoader**: Wrap dataset in PyTorch DataLoader

Model Forward Pass
~~~~~~~~~~~~~~~~~~

1. **Input**: Batch dictionary with keys like ``'encoding.targets'``, ``'decoding.nontargets'``, ``'label.targets'``
2. **Encoding**: ``encode()`` processes encoding chunks → produces context
3. **Decoding**: ``decode()`` uses context to generate predictions → produces outputs dictionary
4. **Output**: Dictionary with keys like ``'head.targets'`` containing predictions

Post-Processing
~~~~~~~~~~~~~~~

1. **Convert to DataFrame**: Use ``ChunkInverter`` to convert tensor outputs to DataFrames
2. **Inverse Transform**: Apply inverse transformation if needed (e.g., denormalize)
3. **Visualize**: Plot predictions and compare with ground truth

6. Model Design Patterns
------------------------

Single-Head Deterministic Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest pattern: one head producing point predictions.

.. code-block:: python

    model = MLP(
        hidden_size=64,
        encoding_length=10,
        decoding_length=5,
        target_names=['target'],
        nontarget_names=[],
        n_hidden_layers=2,
    )
    # Uses default Head with MSE loss

Single-Head Probabilistic Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For uncertainty quantification: one head producing distribution parameters.

.. code-block:: python

    from deep_time_series.core import DistributionHead
    import torch.distributions as dist
    
    head = DistributionHead(
        tag='targets',
        distribution=dist.Normal,
        in_features=64,
        out_features=1,
    )
    
    model = MLP(
        hidden_size=64,
        encoding_length=10,
        decoding_length=5,
        target_names=['target'],
        nontarget_names=[],
        n_hidden_layers=2,
        head=head,
    )

Multi-Head Model
~~~~~~~~~~~~~~~~

For predicting multiple targets or using multiple loss functions.

.. code-block:: python

    from deep_time_series.core import Head
    
    heads = [
        Head(
            tag='target1',
            output_module=nn.Linear(64, 1),
            loss_fn=nn.MSELoss(),
            loss_weight=1.0,
        ),
        Head(
            tag='target2',
            output_module=nn.Linear(64, 1),
            loss_fn=nn.L1Loss(),
            loss_weight=0.5,
        ),
    ]
    
    model.heads = heads

Custom Model
~~~~~~~~~~~~

To create a custom model, inherit from ``ForecastingModule`` and implement:

- ``__init__()``: Set ``encoding_length``, ``decoding_length``, and ``heads``
- ``encode()``: Process encoding chunks and return context
- ``decode_eval()``: Generate predictions from context
- ``make_chunk_specs()``: Define chunk specifications
- ``configure_optimizers()``: Set up optimizer (optional, can use default)
