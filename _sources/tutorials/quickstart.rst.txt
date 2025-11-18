Quick Start Tutorial
=====================

This tutorial provides a quick introduction to DeepTimeSeries.
We'll walk through a complete example from data preparation to model training and prediction.

Installation
------------

First, install DeepTimeSeries:

.. code-block:: bash

    pip install deep-time-series

Or using uv:

.. code-block:: bash

    uv pip install deep-time-series

Basic Example
-------------

Let's start with a simple example: forecasting a sine wave.

Step 1: Prepare Data
~~~~~~~~~~~~~~~~~~~~~

We'll create synthetic time series data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    
    # Generate synthetic data
    time_steps = 200
    t = np.arange(time_steps)
    data = pd.DataFrame({
        'target': np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(time_steps),
        'feature': np.cos(2 * np.pi * t / 50) + 0.1 * np.random.randn(time_steps)
    })
    
    print(data.head())

Step 2: Preprocess Data
~~~~~~~~~~~~~~~~~~~~~~~~

Apply preprocessing to normalize the data:

.. code-block:: python

    import deep_time_series as dts
    from sklearn.preprocessing import StandardScaler
    
    # Create transformer
    transformer = dts.ColumnTransformer(
        transformer_tuples=[
            (StandardScaler(), ['target', 'feature'])
        ]
    )
    
    # Fit and transform
    data_transformed = transformer.fit_transform(data)
    
    print(data_transformed.head())

Step 3: Create Model
~~~~~~~~~~~~~~~~~~~~~

Create an MLP model for forecasting:

.. code-block:: python

    from deep_time_series.model import MLP
    
    model = MLP(
        hidden_size=64,
        encoding_length=20,  # Use last 20 time steps as input
        decoding_length=10,  # Predict next 10 time steps
        target_names=['target'],
        nontarget_names=['feature'],
        n_hidden_layers=2,
        dropout_rate=0.1,
    )
    
    print(model)

Step 4: Create Dataset and DataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a dataset using the model's chunk specifications:

.. code-block:: python

    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = dts.TimeSeriesDataset(
        data_frames=data_transformed,
        chunk_specs=model.make_chunk_specs()
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

Step 5: Train Model
~~~~~~~~~~~~~~~~~~~

Train the model using PyTorch Lightning:

.. code-block:: python

    import pytorch_lightning as pl
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices='auto',
    )
    
    # Train model
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

Step 6: Make Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate predictions on new data:

.. code-block:: python

    import torch
    
    # Get a batch for prediction
    model.eval()
    batch = next(iter(dataloader))
    
    # Generate predictions
    with torch.no_grad():
        predictions = model(batch)
    
    print(f"Prediction keys: {predictions.keys()}")
    print(f"Prediction shape: {predictions['head.targets'].shape}")

Step 7: Convert Predictions to DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert tensor predictions back to DataFrame format:

.. code-block:: python

    from deep_time_series.chunk import ChunkInverter
    
    # Create inverter
    inverter = ChunkInverter(model.make_chunk_specs())
    
    # Convert predictions
    predictions_df = inverter.invert('head.targets', predictions['head.targets'])
    
    print(predictions_df.head())

Complete Example
----------------

Here's the complete code in one place:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    
    import deep_time_series as dts
    from deep_time_series.model import MLP
    from deep_time_series.chunk import ChunkInverter
    
    # 1. Prepare data
    time_steps = 200
    t = np.arange(time_steps)
    data = pd.DataFrame({
        'target': np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(time_steps),
        'feature': np.cos(2 * np.pi * t / 50) + 0.1 * np.random.randn(time_steps)
    })
    
    # 2. Preprocess
    transformer = dts.ColumnTransformer(
        transformer_tuples=[
            (StandardScaler(), ['target', 'feature'])
        ]
    )
    data_transformed = transformer.fit_transform(data)
    
    # 3. Create model
    model = MLP(
        hidden_size=64,
        encoding_length=20,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=['feature'],
        n_hidden_layers=2,
    )
    
    # 4. Create dataset and dataloader
    dataset = dts.TimeSeriesDataset(
        data_frames=data_transformed,
        chunk_specs=model.make_chunk_specs()
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 5. Train
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    
    # 6. Predict
    model.eval()
    batch = next(iter(dataloader))
    with torch.no_grad():
        predictions = model(batch)
    
    # 7. Convert to DataFrame
    inverter = ChunkInverter(model.make_chunk_specs())
    predictions_df = inverter.invert('head.targets', predictions['head.targets'])
    
    print(predictions_df.head())

Next Steps
----------

- Learn about different models in :doc:`models`
- Explore advanced features in :doc:`advanced`
- Read the design guide in :ref:`user_guide:design`

