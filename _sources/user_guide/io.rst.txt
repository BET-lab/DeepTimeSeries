Data Input/Output
=================

This guide explains how to load, preprocess, and transform data in DeepTimeSeries.

Data Format
-----------

DeepTimeSeries uses pandas DataFrame as the primary data format. Each DataFrame represents a time series where:

- Rows correspond to time steps
- Columns correspond to features (target variables and non-target features)
- The index can be used for time information, but is not required

Loading Data
-----------

Since DeepTimeSeries works with pandas DataFrames, you can load data from various sources using pandas:

.. code-block:: python

    import pandas as pd
    
    # Load from CSV
    data = pd.read_csv('data.csv')
    
    # Load from Excel
    data = pd.read_excel('data.xlsx')
    
    # Load from Parquet
    data = pd.read_parquet('data.parquet')
    
    # Create from NumPy array
    import numpy as np
    data = pd.DataFrame({
        'target': np.sin(np.arange(100)),
        'feature1': np.cos(np.arange(100)),
        'feature2': np.random.randn(100)
    })

Multiple Time Series
--------------------

You can work with multiple time series by passing a list of DataFrames to ``TimeSeriesDataset``:

.. code-block:: python

    import pandas as pd
    import deep_time_series as dts
    
    # Create multiple time series
    series1 = pd.DataFrame({'value': np.arange(100)})
    series2 = pd.DataFrame({'value': np.arange(50)})
    
    # Pass as a list
    dataset = dts.TimeSeriesDataset(
        data_frames=[series1, series2],
        chunk_specs=chunk_specs
    )

Data Preprocessing
------------------

The ``ColumnTransformer`` class provides a convenient way to apply transformations to specific columns of your DataFrame.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import deep_time_series as dts
    
    # Create transformer
    transformer = dts.ColumnTransformer(
        transformer_tuples=[
            (StandardScaler(), ['target', 'feature1']),
            (MinMaxScaler(), ['feature2'])
        ]
    )
    
    # Fit and transform
    data_transformed = transformer.fit_transform(data)
    
    # Or fit and transform separately
    transformer.fit(data)
    data_transformed = transformer.transform(data)

Using Transformer Dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use a dictionary to specify transformers:

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    
    transformer = dts.ColumnTransformer(
        transformer_dict={
            'target': StandardScaler(),
            'feature1': StandardScaler(),
            'feature2': StandardScaler()
        }
    )
    
    data_transformed = transformer.fit_transform(data)

Inverse Transform
~~~~~~~~~~~~~~~~~~

To convert transformed data back to original scale:

.. code-block:: python

    # Transform
    data_transformed = transformer.transform(data)
    
    # Inverse transform
    data_original = transformer.inverse_transform(data_transformed)

Understanding Chunk Specifications
-----------------------------------

Chunk specifications define how time series data is extracted and organized for model training and prediction. They are the core concept for understanding how DeepTimeSeries handles temporal data.

What are Chunk Specifications?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A chunk specification (``ChunkSpec``) defines:
- **What features** to extract (via ``names``)
- **When to extract** them (via ``range_``)
- **How to label** them (via ``tag``)
- **What data type** to use (via ``dtype``)

Types of Chunk Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DeepTimeSeries provides three types of chunk specifications:

1. **EncodingChunkSpec**: Defines the input window for the encoder
   - Used for historical data that the model uses to understand patterns
   - Typically has negative range values (e.g., ``range_=(-10, 0)``)

2. **DecodingChunkSpec**: Defines the input window for the decoder
   - Used in autoregressive models where decoder needs previous predictions
   - Can have positive range values (e.g., ``range_=(0, 5)``)

3. **LabelChunkSpec**: Defines the target window for prediction
   - Used for ground truth labels during training
   - Typically has positive range values (e.g., ``range_=(0, 5)``)

Understanding the `range_` Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``range_`` parameter is a tuple ``(start, end)`` that defines the relative time window:

- **Negative values**: Refer to past time steps relative to the current position
- **Zero**: Represents the current time step
- **Positive values**: Refer to future time steps relative to the current position

Examples:

.. code-block:: python

    import numpy as np
    import deep_time_series as dts
    
    # Extract 10 past time steps (t-10 to t-1, excluding current time)
    encoding_spec = dts.EncodingChunkSpec(
        tag='input',
        names=['target', 'feature1'],
        range_=(-10, 0),  # From 10 steps ago to current (exclusive)
        dtype=np.float32
    )
    
    # Extract 5 future time steps (t to t+4)
    label_spec = dts.LabelChunkSpec(
        tag='output',
        names=['target'],
        range_=(0, 5),  # From current to 4 steps ahead
        dtype=np.float32
    )
    
    # Extract overlapping window (t-5 to t+5)
    window_spec = dts.EncodingChunkSpec(
        tag='context',
        names=['target'],
        range_=(-5, 6),  # 5 steps before to 5 steps after
        dtype=np.float32
    )

Visual Representation
~~~~~~~~~~~~~~~~~~~~~~

For a time series with data points at indices 0, 1, 2, ..., and a chunk specification with ``range_=(-3, 2)``:

::

    Time indices:  ...  -3  -2  -1   0   1   2   3  ...
    Relative to t: ...  -3  -2  -1  [t]  1   2   3  ...
    
    range_=(-3, 2) extracts: [t-3, t-2, t-1, t, t+1]
                              (5 time steps total)

Tag Naming Convention
~~~~~~~~~~~~~~~~~~~~~

Tags are used to identify different chunks in the data dictionary. They follow a prefix pattern:

- ``EncodingChunkSpec``: Automatically adds ``'encoding.'`` prefix
- ``DecodingChunkSpec``: Automatically adds ``'decoding.'`` prefix  
- ``LabelChunkSpec``: Automatically adds ``'label.'`` prefix

Example:

.. code-block:: python

    spec = dts.EncodingChunkSpec(tag='my_feature', ...)
    # spec.tag will be 'encoding.my_feature'
    
    spec = dts.LabelChunkSpec(tag='target', ...)
    # spec.tag will be 'label.target'

Tags must be unique within a list of chunk specifications. When you access data from ``TimeSeriesDataset``, you'll use these tags as dictionary keys.

Choosing the Right Data Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``dtype`` parameter controls the NumPy data type used for the extracted chunks:

- **np.float32**: Recommended for most cases (memory efficient, sufficient precision)
- **np.float64**: Use when high precision is required (uses more memory)
- **np.int32**: Use for integer features (e.g., counts, categories)

Example:

.. code-block:: python

    # For continuous values
    continuous_spec = dts.EncodingChunkSpec(
        tag='features',
        names=['temperature', 'humidity'],
        range_=(-10, 0),
        dtype=np.float32  # Standard for neural networks
    )
    
    # For integer counts
    count_spec = dts.EncodingChunkSpec(
        tag='counts',
        names=['sales_count', 'visitor_count'],
        range_=(-10, 0),
        dtype=np.int32
    )

Creating Dataset
----------------

After preprocessing, create a ``TimeSeriesDataset`` with appropriate chunk specifications:

.. code-block:: python

    import deep_time_series as dts
    from deep_time_series.chunk import EncodingChunkSpec, LabelChunkSpec
    import numpy as np
    
    # Define chunk specifications
    encoding_spec = dts.EncodingChunkSpec(
        tag='input',
        names=['target', 'feature1', 'feature2'],
        range_=(-10, 0),  # 10 time steps before current time
        dtype=np.float32
    )
    
    label_spec = dts.LabelChunkSpec(
        tag='output',
        names=['target'],
        range_=(0, 5),  # 5 time steps ahead
        dtype=np.float32
    )
    
    # Create dataset
    dataset = dts.TimeSeriesDataset(
        data_frames=data_transformed,
        chunk_specs=[encoding_spec, label_spec],
        return_time_index=True  # Optional: include time index in chunks
    )

How ChunkExtractor Works
-------------------------

The ``ChunkExtractor`` class is responsible for extracting data chunks from a DataFrame according to chunk specifications. Understanding its behavior helps you debug and optimize your data pipeline.

Data Extraction Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~

When you create a ``TimeSeriesDataset``, it internally creates a ``ChunkExtractor`` for each DataFrame. The extractor:

1. **Preprocesses the DataFrame**: Converts specified columns to the required data type
2. **Calculates window boundaries**: Determines the minimum and maximum time indices needed
3. **Extracts chunks**: For each sample, extracts the specified time windows

Example of extraction process:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from deep_time_series.chunk import ChunkExtractor, EncodingChunkSpec, LabelChunkSpec
    
    # Create sample data
    df = pd.DataFrame({
        'target': np.arange(20),
        'feature': np.arange(20) * 2
    })
    
    # Define specifications
    encoding_spec = EncodingChunkSpec(
        tag='input',
        names=['target', 'feature'],
        range_=(-3, 0),  # Last 3 time steps
        dtype=np.float32
    )
    
    label_spec = LabelChunkSpec(
        tag='output',
        names=['target'],
        range_=(0, 2),  # Next 2 time steps
        dtype=np.float32
    )
    
    # Create extractor
    extractor = ChunkExtractor(df, [encoding_spec, label_spec])
    
    # Extract chunk at time index 5
    chunk = extractor.extract(start_time_index=5, return_time_index=False)
    
    # chunk contains:
    # - 'encoding.input': shape (3, 2) - 3 time steps, 2 features
    # - 'label.output': shape (2, 1) - 2 time steps, 1 feature

Understanding `start_time_index`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``start_time_index`` parameter determines where in the time series to start extraction:

- **start_time_index = 0**: Extract from the beginning of the series
- **start_time_index = 5**: Extract starting from index 5
- The extractor ensures that ``start_time_index + chunk_min_t >= 0``

For the example above with ``range_=(-3, 0)`` for encoding:
- At ``start_time_index=5``, it extracts indices [2, 3, 4] (5-3 to 5-1)
- At ``start_time_index=0``, it cannot extract because 0 + (-3) < 0

Multiple Chunk Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple chunk specifications are provided, the extractor:

1. Finds the overall window: ``[min(range[0]), max(range[1])]``
2. Extracts data for the entire window
3. Then slices each specification's portion

Example:

.. code-block:: python

    # Specification 1: range_=(-5, 0)
    # Specification 2: range_=(0, 3)
    # Overall window: range_=(-5, 3) - extracts 8 time steps total
    
    spec1 = EncodingChunkSpec(tag='past', names=['target'], range_=(-5, 0), dtype=np.float32)
    spec2 = LabelChunkSpec(tag='future', names=['target'], range_=(0, 3), dtype=np.float32)
    
    extractor = ChunkExtractor(df, [spec1, spec2])
    # chunk_min_t = -5, chunk_max_t = 3, chunk_length = 8
    
    chunk = extractor.extract(start_time_index=10)
    # Extracts indices [5, 6, 7, 8, 9, 10, 11, 12]
    # Then slices:
    #   - 'encoding.past': indices [5, 6, 7, 8, 9] (relative -5 to -1)
    #   - 'label.future': indices [10, 11, 12] (relative 0 to 2)

The `return_time_index` Option
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``return_time_index=True``, the extractor also includes time index information:

.. code-block:: python

    chunk = extractor.extract(start_time_index=5, return_time_index=True)
    
    # chunk contains:
    # - 'encoding.input': the data array
    # - 'encoding.input.time_index': array([2, 3, 4]) - actual DataFrame indices
    # - 'label.output': the data array
    # - 'label.output.time_index': array([5, 6]) - actual DataFrame indices

This is useful when you need to track which time points correspond to predictions.

TimeSeriesDataset Deep Dive
----------------------------

The ``TimeSeriesDataset`` class wraps ``ChunkExtractor`` to provide a PyTorch-compatible Dataset interface.

Dataset Length Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The length of the dataset depends on:

1. **DataFrame length**: Total number of time steps
2. **Chunk length**: Maximum window size needed (``chunk_max_t - chunk_min_t``)
3. **Minimum start index**: Ensures valid extractions (``max(0, -chunk_min_t)``)

Formula: ``length = len(df) - chunk_length + 1 - min_start_time_index``

Example:

.. code-block:: python

    df = pd.DataFrame({'value': np.arange(100)})  # 100 time steps
    
    spec = EncodingChunkSpec(
        tag='input',
        names=['value'],
        range_=(-10, 0),  # chunk_length = 10
        dtype=np.float32
    )
    
    dataset = TimeSeriesDataset(df, [spec])
    # Length = 100 - 10 + 1 - 0 = 91 samples
    
    # Can extract from indices 0 to 90
    # At index 0: extracts [0:10]
    # At index 90: extracts [90:100]

Multiple DataFrames Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple DataFrames are provided, the dataset:

1. Creates a separate ``ChunkExtractor`` for each DataFrame
2. Concatenates samples from all DataFrames
3. Uses cumulative indexing to map dataset index to DataFrame index

Example:

.. code-block:: python

    series1 = pd.DataFrame({'value': np.arange(50)})   # 50 time steps
    series2 = pd.DataFrame({'value': np.arange(30)})   # 30 time steps
    
    dataset = TimeSeriesDataset([series1, series2], chunk_specs)
    
    # series1 contributes: 50 - chunk_length + 1 samples
    # series2 contributes: 30 - chunk_length + 1 samples
    # Total length = sum of contributions
    
    # dataset[0] to dataset[N-1]: from series1
    # dataset[N] to dataset[M-1]: from series2

Indexing Behavior
~~~~~~~~~~~~~~~~~

The ``__getitem__`` method:

1. Determines which DataFrame the index belongs to using cumulative sum
2. Calculates the relative index within that DataFrame
3. Adjusts for ``min_start_time_index`` to get the actual start position
4. Calls the appropriate ``ChunkExtractor.extract()``

Memory Efficiency
~~~~~~~~~~~~~~~~~

The dataset stores:
- **DataFrames**: Original data (can be large)
- **ChunkExtractors**: Lightweight objects that reference the DataFrames
- **Metadata**: Lengths, indices (minimal memory)

Data is extracted on-demand during ``__getitem__``, so memory usage is efficient. However, for very large datasets, consider:

- Using ``DataLoader`` with ``num_workers > 0`` for parallel loading
- Preprocessing and saving to disk in chunk format
- Using data streaming for extremely large datasets

Model-Based Chunk Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models can automatically generate chunk specifications:

.. code-block:: python

    from deep_time_series.model import MLP
    
    # Create model
    model = MLP(
        encoding_length=10,
        decoding_length=5,
        target_names=['target'],
        nontarget_names=['feature1', 'feature2'],
        # ... other parameters
    )
    
    # Get chunk specifications from model
    chunk_specs = model.make_chunk_specs()
    
    # Create dataset
    dataset = dts.TimeSeriesDataset(
        data_frames=data_transformed,
        chunk_specs=chunk_specs
    )

The ``make_chunk_specs()`` method creates appropriate specifications based on:
- ``encoding_length``: Creates ``EncodingChunkSpec`` with ``range_=(-encoding_length, 0)``
- ``decoding_length``: Creates ``DecodingChunkSpec`` and ``LabelChunkSpec`` with ``range_=(0, decoding_length)``
- ``target_names`` and ``nontarget_names``: Determines which features to include

Converting Model Outputs
------------------------

After model prediction, use ``ChunkInverter`` to convert tensor outputs back to DataFrame format:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import torch
    from deep_time_series.chunk import ChunkInverter
    
    # Create inverter (use the same chunk_specs as dataset)
    inverter = ChunkInverter(chunk_specs)
    
    # Model outputs are dictionaries with tag keys
    # Example output from model
    outputs = {
        'head.target': torch.randn(32, 5, 1)  # (batch, time, features)
    }
    
    # Convert to DataFrame
    df_output = inverter.invert('head.target', outputs['head.target'])
    # Returns DataFrame with columns ['target'] and MultiIndex (batch_index, time_index)

Converting Multiple Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Convert entire output dictionary
    outputs_dict = inverter.invert_dict(outputs)
    # Returns dictionary of DataFrames

Tag Matching
~~~~~~~~~~~~

The inverter can match tags in different formats:

- Full tag: ``'head.target'``
- Core tag: ``'target'``
- Tag without prefix: ``'label.target'`` → matches ``'target'``

Practical Examples
------------------

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Full example from data loading to prediction:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    
    import deep_time_series as dts
    from deep_time_series.model import MLP
    from deep_time_series.chunk import ChunkInverter
    
    # 1. Load data
    data = pd.read_csv('timeseries_data.csv')
    
    # 2. Preprocess
    transformer = dts.ColumnTransformer(
        transformer_tuples=[
            (StandardScaler(), data.columns)
        ]
    )
    data_transformed = transformer.fit_transform(data)
    
    # 3. Create model
    model = MLP(
        hidden_size=64,
        encoding_length=10,
        decoding_length=5,
        target_names=['target'],
        nontarget_names=['feature1', 'feature2'],
        n_hidden_layers=2,
    )
    
    # 4. Create dataset
    dataset = dts.TimeSeriesDataset(
        data_frames=data_transformed,
        chunk_specs=model.make_chunk_specs()
    )
    dataloader = DataLoader(dataset, batch_size=32)
    
    # 5. Train model
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=dataloader)
    
    # 6. Make predictions
    model.eval()
    batch = next(iter(dataloader))
    with torch.no_grad():
        predictions = model(batch)
    
    # 7. Convert predictions to DataFrame
    inverter = ChunkInverter(model.make_chunk_specs())
    predictions_df = inverter.invert('head.target', predictions['head.target'])
    
    # 8. Inverse transform if needed
    predictions_original_scale = transformer.inverse_transform(
        predictions_df.reset_index(level='time_index', drop=True)
    )

Example: Multiple Time Series with Different Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with multiple time series that need different preprocessing:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import deep_time_series as dts
    
    # Load multiple series
    series1 = pd.read_csv('series1.csv')
    series2 = pd.read_csv('series2.csv')
    
    # Different transformers for different series
    transformer1 = dts.ColumnTransformer(
        transformer_tuples=[(StandardScaler(), series1.columns)]
    )
    transformer2 = dts.ColumnTransformer(
        transformer_tuples=[(RobustScaler(), series2.columns)]
    )
    
    # Transform separately
    series1_transformed = transformer1.fit_transform(series1)
    series2_transformed = transformer2.fit_transform(series2)
    
    # Create model
    model = dts.model.MLP(
        encoding_length=20,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=['feature1', 'feature2'],
        # ... other parameters
    )
    
    # Create dataset with multiple series
    dataset = dts.TimeSeriesDataset(
        data_frames=[series1_transformed, series2_transformed],
        chunk_specs=model.make_chunk_specs()
    )

Example: Custom Chunk Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating custom chunk specifications for advanced use cases:

.. code-block:: python

    import numpy as np
    import deep_time_series as dts
    from deep_time_series.chunk import (
        EncodingChunkSpec,
        DecodingChunkSpec,
        LabelChunkSpec
    )
    
    # Define custom specifications
    # Encoding: last 30 days
    encoding_spec = EncodingChunkSpec(
        tag='historical',
        names=['sales', 'price', 'promotion'],
        range_=(-30, 0),
        dtype=np.float32
    )
    
    # Decoding: autoregressive input (previous predictions)
    decoding_spec = DecodingChunkSpec(
        tag='previous_pred',
        names=['sales'],
        range_=(-5, 0),  # Last 5 predictions
        dtype=np.float32
    )
    
    # Label: next 7 days
    label_spec = LabelChunkSpec(
        tag='forecast',
        names=['sales'],
        range_=(0, 7),
        dtype=np.float32
    )
    
    # Create dataset
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=[encoding_spec, decoding_spec, label_spec]
    )
    
    # Access chunks
    sample = dataset[0]
    # sample['encoding.historical']: shape (30, 3)
    # sample['decoding.previous_pred']: shape (5, 1)
    # sample['label.forecast']: shape (7, 1)

Example: Working with Time Index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using time index information for tracking predictions:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import deep_time_series as dts
    
    # Create data with datetime index
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'value': np.random.randn(100)
    }, index=dates)
    
    # Create dataset with return_time_index=True
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=chunk_specs,
        return_time_index=True
    )
    
    # Get a sample
    sample = dataset[10]
    
    # Access time indices
    encoding_times = sample['encoding.input.time_index']
    label_times = sample['label.output.time_index']
    
    # Convert to actual datetime if needed
    encoding_dates = dates[encoding_times]
    label_dates = dates[label_times]
    
    print(f"Encoding period: {encoding_dates[0]} to {encoding_dates[-1]}")
    print(f"Forecast period: {label_dates[0]} to {label_dates[-1]}")

Example: Handling Missing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocessing data with missing values:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import deep_time_series as dts
    
    # Load data with missing values
    data = pd.read_csv('data_with_missing.csv')
    
    # Handle missing values before creating transformer
    # Option 1: Forward fill
    data_ffill = data.fillna(method='ffill')
    
    # Option 2: Interpolation
    data_interp = data.interpolate(method='linear')
    
    # Option 3: Drop rows with missing values
    data_clean = data.dropna()
    
    # Then apply transformer
    transformer = dts.ColumnTransformer(
        transformer_tuples=[(StandardScaler(), data_clean.columns)]
    )
    data_transformed = transformer.fit_transform(data_clean)
    
    # Create dataset
    dataset = dts.TimeSeriesDataset(
        data_frames=data_transformed,
        chunk_specs=chunk_specs
    )

Example: Visualizing Chunk Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualizing how chunks are extracted:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import deep_time_series as dts
    
    # Create sample data
    data = pd.DataFrame({
        'value': np.sin(np.arange(100) * 0.1)
    })
    
    # Create specifications
    encoding_spec = dts.EncodingChunkSpec(
        tag='input',
        names=['value'],
        range_=(-10, 0),
        dtype=np.float32
    )
    
    label_spec = dts.LabelChunkSpec(
        tag='output',
        names=['value'],
        range_=(0, 5),
        dtype=np.float32
    )
    
    # Create dataset
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=[encoding_spec, label_spec]
    )
    
    # Visualize chunk structure
    dataset.plot_chunks()  # Shows the chunk ranges
    
    # Get a sample and visualize
    sample = dataset[50]
    
    plt.figure(figsize=(12, 4))
    plt.plot(data['value'], label='Full series', alpha=0.3)
    
    # Plot encoding window
    encoding_data = sample['encoding.input']
    encoding_start = 50 - 10
    plt.plot(range(encoding_start, encoding_start + 10), 
             encoding_data.flatten(), 'o-', label='Encoding window')
    
    # Plot label window
    label_data = sample['label.output']
    plt.plot(range(50, 50 + 5), 
             label_data.flatten(), 's-', label='Label window')
    
    plt.legend()
    plt.title('Chunk Extraction Visualization')
    plt.show()

Troubleshooting Common Issues
-------------------------------

This section covers common errors and how to resolve them.

Error: "Tags are duplicated"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: You get a ``ValueError`` saying tags are duplicated.

**Cause**: Multiple chunk specifications have the same tag.

**Solution**: Ensure each chunk specification has a unique tag:

.. code-block:: python

    # Wrong: Same tag used twice
    spec1 = EncodingChunkSpec(tag='input', ...)
    spec2 = EncodingChunkSpec(tag='input', ...)  # Error!
    
    # Correct: Different tags
    spec1 = EncodingChunkSpec(tag='features', ...)
    spec2 = EncodingChunkSpec(tag='targets', ...)

Error: "range[0] >= range[1]"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: You get a ``ValueError`` about invalid range.

**Cause**: The ``range_`` tuple has ``start >= end``.

**Solution**: Ensure ``range_[0] < range_[1]``:

.. code-block:: python

    # Wrong
    spec = EncodingChunkSpec(tag='input', range_=(5, 3), ...)  # Error!
    
    # Correct
    spec = EncodingChunkSpec(tag='input', range_=(-5, 3), ...)

Error: AssertionError in ChunkExtractor.extract()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ``AssertionError`` when extracting chunks: ``start_time_index + chunk_min_t >= 0``.

**Cause**: Trying to extract a chunk that requires data before index 0.

**Solution**: Ensure your data is long enough or adjust the range:

.. code-block:: python

    # If you have 100 time steps and range_=(-20, 0)
    # You can only extract from index 20 onwards
    
    # Option 1: Use longer data
    # Option 2: Reduce the encoding window
    spec = EncodingChunkSpec(tag='input', range_=(-10, 0), ...)  # Smaller window
    
    # Option 3: Pad your data at the beginning
    # (Not recommended, consider using range_ that doesn't go negative)

Error: Column not found in DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ``KeyError`` when creating chunk specifications.

**Cause**: Column names in ``names`` parameter don't match DataFrame columns.

**Solution**: Verify column names match exactly:

.. code-block:: python

    # Check available columns
    print(data.columns.tolist())
    
    # Ensure names match exactly (case-sensitive)
    spec = EncodingChunkSpec(
        tag='input',
        names=['target', 'feature1'],  # Must match DataFrame columns exactly
        ...
    )

Error: Dataset length is 0 or too small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Dataset has fewer samples than expected.

**Cause**: Data is too short for the specified chunk ranges.

**Solution**: Check data length and chunk requirements:

.. code-block:: python

    # Check data length
    print(f"Data length: {len(data)}")
    
    # Check chunk length requirement
    chunk_specs = model.make_chunk_specs()
    min_required = max(-spec.range[0] for spec in chunk_specs)
    print(f"Minimum data length needed: {min_required}")
    
    # Ensure: len(data) >= min_required + decoding_length

Error: Shape mismatch in model forward pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Model receives data with unexpected shape.

**Cause**: Chunk specifications don't match model expectations.

**Solution**: Use model's ``make_chunk_specs()`` method:

.. code-block:: python

    # Always use model's chunk specs
    chunk_specs = model.make_chunk_specs()
    dataset = TimeSeriesDataset(data, chunk_specs=chunk_specs)
    
    # Don't manually create specs unless you know what you're doing

Error: ChunkInverter cannot find tag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ``ValueError``: tag not found in chunk_specs when using ``ChunkInverter``.

**Cause**: Tag doesn't match any chunk specification.

**Solution**: Use correct tag format:

.. code-block:: python

    # Check available tags
    tags = [spec.tag for spec in chunk_specs]
    print(f"Available tags: {tags}")
    
    # Use full tag or core tag
    inverter.invert('head.target', tensor)  # Full tag
    inverter.invert('target', tensor)        # Core tag (also works)

Performance Issues
~~~~~~~~~~~~~~~~~~

**Problem**: Data loading is slow.

**Solutions**:

1. **Use DataLoader with multiple workers**:

   .. code-block:: python

       dataloader = DataLoader(
           dataset, 
           batch_size=32,
           num_workers=4,  # Parallel data loading
           pin_memory=True  # Faster GPU transfer
       )

2. **Preprocess data once and save**:

   .. code-block:: python

       # Preprocess and save
       data_transformed = transformer.fit_transform(data)
       data_transformed.to_parquet('preprocessed_data.parquet')
       
       # Load preprocessed data later
       data_transformed = pd.read_parquet('preprocessed_data.parquet')

3. **Reduce chunk window size** if possible (smaller windows = faster extraction)

Data Type Issues
~~~~~~~~~~~~~~~~

**Problem**: Unexpected data types or precision loss.

**Solution**: Explicitly set dtype in chunk specifications:

.. code-block:: python

    # For neural networks, use float32 (standard)
    spec = EncodingChunkSpec(
        tag='input',
        names=['value'],
        range_=(-10, 0),
        dtype=np.float32  # Explicitly set
    )
    
    # For high precision requirements
    spec = EncodingChunkSpec(
        tag='input',
        names=['value'],
        range_=(-10, 0),
        dtype=np.float64  # Higher precision, more memory
    )

Tips and Best Practices
-----------------------

1. **Data Validation**: 
   - Ensure your DataFrame has no missing values before creating the dataset
   - Check data types are appropriate (numeric for most models)
   - Verify column names match chunk specification names

2. **Index Handling**: 
   - The dataset uses integer indices internally
   - If your DataFrame has a datetime index, store it separately or reset it
   - Use ``return_time_index=True`` if you need to track time points

3. **Memory Efficiency**: 
   - For large datasets, use ``DataLoader`` with appropriate ``num_workers``
   - Consider preprocessing and saving to disk
   - Use ``float32`` instead of ``float64`` when possible

4. **Preprocessing**: 
   - Always fit transformers on training data only
   - Transform both training and validation/test data with the same transformer
   - Save transformers for inference time

5. **Chunk Specifications**: 
   - Use model's ``make_chunk_specs()`` when possible
   - Ensure chunk ranges don't exceed your data length
   - Consider the minimum data length requirement: ``len(data) >= max(-range[0]) + max(range[1])``

6. **Debugging**: 
   - Use ``dataset.plot_chunks()`` to visualize chunk structure
   - Inspect a single sample: ``sample = dataset[0]``
   - Check shapes: ``print({k: v.shape for k, v in sample.items()})``
   - Verify time indices if using ``return_time_index=True``

7. **Multiple Time Series**: 
   - Ensure all DataFrames have the same column structure
   - Apply consistent preprocessing to all series
   - Consider series length differences when interpreting results

