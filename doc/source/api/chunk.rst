Chunk Specification
===================

The chunk module provides classes for defining and extracting chunks from time series data. Chunks are used to specify the input windows for encoding, decoding, and the target windows for prediction.

Understanding Chunks
--------------------

In DeepTimeSeries, a **chunk** represents a time window within a time series. The framework uses three types of chunks:

1. **Encoding Chunks**: Input features used by the encoder (historical data)
2. **Decoding Chunks**: Input features used by the decoder during autoregressive prediction (future known features)
3. **Label Chunks**: Target values for prediction (what we want to predict)

**Time Range Convention:**

Chunks use relative time indices where:
- Time ``0`` represents the current time step
- Negative indices represent past time steps (e.g., ``-10`` is 10 steps ago)
- Positive indices represent future time steps (e.g., ``5`` is 5 steps ahead)

**Example:**

For a model with encoding_length=10 and decoding_length=5:

- Encoding window: ``(0, 10)`` - uses time steps 0 to 9 (current and past)
- Decoding window: ``(10, 15)`` - uses time steps 10 to 14 (future known features)
- Label window: ``(10, 15)`` - predicts time steps 10 to 14

.. automodule:: deep_time_series.chunk
   :members:
   :undoc-members:
   :show-inheritance:

BaseChunkSpec
~~~~~~~~~~~~~

Base class for all chunk specifications. Defines the structure for specifying time windows and feature names.

**Properties:**

- ``tag``: Unique identifier for the chunk (e.g., ``'encoding.targets'``, ``'decoding.nontargets'``)
- ``names``: List of column names from the DataFrame to include in this chunk
- ``range_``: Tuple ``(start, end)`` defining the time window (relative indices)
- ``dtype``: NumPy dtype for the data (typically ``np.float32``)

**Tag Prefixes:**

Each chunk type has an automatic prefix:
- ``EncodingChunkSpec``: ``'encoding'``
- ``DecodingChunkSpec``: ``'decoding'``
- ``LabelChunkSpec``: ``'label'``

**Example:**

.. code-block:: python

   from deep_time_series.chunk import EncodingChunkSpec
   import numpy as np

   spec = EncodingChunkSpec(
       tag='targets',
       names=['temperature', 'humidity'],
       range_=(0, 10),  # Use time steps 0-9
       dtype=np.float32,
   )
   # spec.tag will be 'encoding.targets'

.. autoclass:: deep_time_series.chunk.BaseChunkSpec
   :members:
   :undoc-members:
   :show-inheritance:

EncodingChunkSpec
~~~~~~~~~~~~~~~~~

Specification for the encoding window, which defines the input features for the encoder.

**Use Case:**

Define what historical data the model should use to make predictions. This typically includes:
- Target features (what you're predicting)
- Non-target features (exogenous variables)

**Time Range:**

Usually starts at time ``0`` and extends backward or forward. For example:
- ``(0, 10)``: Uses current time step and 9 past steps
- ``(-10, 0)``: Uses 10 past steps (excluding current)

**Example:**

.. code-block:: python

   from deep_time_series.chunk import EncodingChunkSpec
   import numpy as np

   # Encode using last 20 time steps of temperature and pressure
   encoding_spec = EncodingChunkSpec(
       tag='targets',
       names=['temperature', 'pressure'],
       range_=(0, 20),
       dtype=np.float32,
   )

.. autoclass:: deep_time_series.chunk.EncodingChunkSpec
   :members:
   :undoc-members:
   :show-inheritance:

DecodingChunkSpec
~~~~~~~~~~~~~~~~~

Specification for the decoding window, which defines the input features for the decoder during autoregressive prediction.

**Use Case:**

Define future known features that the decoder can use during autoregressive prediction. This is useful when you have exogenous variables (non-target features) that are known in advance.

**Time Range:**

Typically starts after the encoding window and extends into the future. For example:
- ``(10, 15)``: Uses time steps 10-14 (5 future steps)

**Important:**

- Only non-target features should be specified here (targets are predicted, not provided)
- If you don't have future known features, you don't need a ``DecodingChunkSpec``

**Example:**

.. code-block:: python

   from deep_time_series.chunk import DecodingChunkSpec
   import numpy as np

   # Use future known features (e.g., scheduled events, weather forecasts)
   decoding_spec = DecodingChunkSpec(
       tag='nontargets',
       names=['scheduled_events', 'weather_forecast'],
       range_=(10, 15),  # Future time steps 10-14
       dtype=np.float32,
   )

.. autoclass:: deep_time_series.chunk.DecodingChunkSpec
   :members:
   :undoc-members:
   :show-inheritance:

LabelChunkSpec
~~~~~~~~~~~~~~

Specification for the label window, which defines the target values for prediction.

**Use Case:**

Define what the model should predict. This is used during training to compute loss and during evaluation to compare predictions with ground truth.

**Time Range:**

Typically matches the prediction horizon. For example:
- ``(10, 15)``: Predict time steps 10-14 (5 steps ahead)

**Important:**

- The label window should align with your prediction horizon
- The names should match the target features you want to predict

**Example:**

.. code-block:: python

   from deep_time_series.chunk import LabelChunkSpec
   import numpy as np

   # Predict next 5 time steps of temperature
   label_spec = LabelChunkSpec(
       tag='targets',
       names=['temperature'],
       range_=(10, 15),  # Predict time steps 10-14
       dtype=np.float32,
   )

.. autoclass:: deep_time_series.chunk.LabelChunkSpec
   :members:
   :undoc-members:
   :show-inheritance:

ChunkExtractor
~~~~~~~~~~~~~~

Utility class for extracting chunks from pandas DataFrames based on chunk specifications.

**Purpose:**

``ChunkExtractor`` takes a DataFrame and a list of chunk specifications, then extracts the appropriate time windows when given a starting time index.

**How It Works:**

1. **Initialization**: Preprocesses the DataFrame and validates chunk specifications
2. **Extraction**: Given a start time index, extracts all specified chunks
3. **Output**: Returns a dictionary with chunk tags as keys and numpy arrays as values

**Key Properties:**

- ``chunk_min_t``: Minimum time index across all chunks
- ``chunk_max_t``: Maximum time index across all chunks
- ``chunk_length``: Total length of the chunk window

**Example:**

.. code-block:: python

   import pandas as pd
   from deep_time_series.chunk import ChunkExtractor, EncodingChunkSpec, LabelChunkSpec
   import numpy as np

   # Create sample data
   df = pd.DataFrame({
       'temperature': np.random.randn(100),
       'humidity': np.random.randn(100),
   })

   # Define chunk specifications
   chunk_specs = [
       EncodingChunkSpec('targets', ['temperature'], (0, 10), np.float32),
       LabelChunkSpec('targets', ['temperature'], (10, 15), np.float32),
   ]

   # Create extractor
   extractor = ChunkExtractor(df, chunk_specs)

   # Extract chunks starting at time index 20
   chunks = extractor.extract(start_time_index=20, return_time_index=True)
   # Returns: {
   #   'encoding.targets': array of shape (10, 1),
   #   'label.targets': array of shape (5, 1),
   #   'encoding.targets.time_index': array([20, 21, ..., 29]),
   #   'label.targets.time_index': array([30, 31, ..., 34]),
   # }

**Note:**

The extractor ensures that all chunks are extracted from the same time window, maintaining temporal alignment.

.. autoclass:: deep_time_series.chunk.ChunkExtractor
   :members:
   :undoc-members:
   :show-inheritance:

ChunkInverter
~~~~~~~~~~~~~

Utility class for converting model outputs (tensors) back to pandas DataFrames.

**Purpose:**

After model inference, outputs are PyTorch tensors. ``ChunkInverter`` converts these tensors back to pandas DataFrames with proper column names and structure.

**How It Works:**

1. **Initialization**: Takes chunk specifications to understand the output structure
2. **Inversion**: Converts tensors (or numpy arrays) to DataFrames using the chunk tag and feature names
3. **Output**: Returns DataFrames with proper column names matching the original data

**Input Format:**

Tensors should have shape ``(batch_size, time_steps, n_features)`` or ``(time_steps, n_features)`` for single samples.

**Example:**

.. code-block:: python

   import torch
   from deep_time_series.chunk import ChunkInverter, LabelChunkSpec
   import numpy as np

   # Define chunk specification
   chunk_specs = [
       LabelChunkSpec('targets', ['temperature', 'humidity'], (10, 15), np.float32),
   ]

   # Create inverter
   inverter = ChunkInverter(chunk_specs)

   # Model output (batch_size=32, time_steps=5, n_features=2)
   outputs = torch.randn(32, 5, 2)

   # Convert to DataFrame
   df = inverter.invert('label.targets', outputs)
   # Returns DataFrame with columns ['temperature', 'humidity']
   # and MultiIndex with 'batch_index' and 'time_index'

**Tag Matching:**

The inverter can match tags in several ways:
- Full tag: ``'label.targets'``
- Core tag: ``'targets'`` (without prefix)
- Partial tag: ``'head.targets'`` (extracts core tag)

**Use Cases:**

- Converting model predictions to DataFrames for analysis
- Saving predictions in a readable format
- Comparing predictions with ground truth DataFrames

.. autoclass:: deep_time_series.chunk.ChunkInverter
   :members:
   :undoc-members:
   :show-inheritance:
