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

Creating Dataset
----------------

After preprocessing, create a ``TimeSeriesDataset`` with appropriate chunk specifications:

.. code-block:: python

    import deep_time_series as dts
    from deep_time_series.chunk import EncodingChunkSpec, LabelChunkSpec
    
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

Example Workflow
----------------

Complete example from data loading to prediction:

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

Tips and Best Practices
-----------------------

1. **Data Validation**: Ensure your DataFrame has no missing values before creating the dataset
2. **Index Handling**: The dataset uses integer indices internally. If your DataFrame has a time index, consider resetting it or storing it separately
3. **Memory Efficiency**: For large datasets, consider using ``DataLoader`` with appropriate ``num_workers``
4. **Preprocessing**: Always fit transformers on training data only, then transform both training and validation/test data
5. **Chunk Specifications**: Make sure chunk ranges don't exceed your data length, especially for the encoding window

