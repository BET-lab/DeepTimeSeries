Model Tutorials
================

This tutorial covers the different models available in DeepTimeSeries and how to use them.

All models in DeepTimeSeries inherit from ``ForecastingModule`` and follow the same interface,
making it easy to switch between different architectures.

MLP Model
---------

The MLP (Multi-Layer Perceptron) model is a simple feedforward neural network that flattens
the encoding window and processes it through fully connected layers.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    
    import deep_time_series as dts
    from deep_time_series.model import MLP
    
    # Prepare data
    data = pd.DataFrame({
        'target': np.sin(np.arange(100)),
        'feature': np.cos(np.arange(100))
    })
    
    # Preprocess
    transformer = dts.ColumnTransformer(
        transformer_tuples=[(StandardScaler(), ['target', 'feature'])]
    )
    data = transformer.fit_transform(data)
    
    # Create MLP model
    model = MLP(
        hidden_size=64,
        encoding_length=10,
        decoding_length=5,
        target_names=['target'],
        nontarget_names=['feature'],
        n_hidden_layers=2,
        activation=torch.nn.ELU,
        dropout_rate=0.1,
    )
    
    # Create dataset and train
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=model.make_chunk_specs()
    )
    dataloader = DataLoader(dataset, batch_size=32)
    
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=dataloader)

Parameters
~~~~~~~~~~

- ``hidden_size``: Size of hidden layers
- ``encoding_length``: Length of encoding window
- ``decoding_length``: Length of decoding window
- ``target_names``: List of target feature names
- ``nontarget_names``: List of non-target feature names
- ``n_hidden_layers``: Number of hidden layers
- ``activation``: Activation function class (default: ``nn.ELU``)
- ``dropout_rate``: Dropout rate (default: 0.0)

RNN Models
----------

The RNN model supports vanilla RNN, LSTM, and GRU architectures.
It uses a recurrent encoder and decoder for sequential processing.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import torch.nn as nn
    from deep_time_series.model import RNN
    
    # Create LSTM model
    model = RNN(
        hidden_size=128,
        encoding_length=20,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=['feature'],
        n_layers=2,
        rnn_class=nn.LSTM,  # or nn.RNN, nn.GRU
        dropout_rate=0.1,
    )
    
    # Use the same way as MLP
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=model.make_chunk_specs()
    )
    dataloader = DataLoader(dataset, batch_size=32)
    
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=dataloader)

RNN Variants
~~~~~~~~~~~~

You can use different RNN variants:

.. code-block:: python

    # Vanilla RNN
    model_rnn = RNN(..., rnn_class=nn.RNN)
    
    # LSTM
    model_lstm = RNN(..., rnn_class=nn.LSTM)
    
    # GRU
    model_gru = RNN(..., rnn_class=nn.GRU)

Parameters
~~~~~~~~~~

- ``hidden_size``: Hidden state size
- ``encoding_length``: Length of encoding window
- ``decoding_length``: Length of decoding window
- ``target_names``: List of target feature names
- ``nontarget_names``: List of non-target feature names
- ``n_layers``: Number of RNN layers
- ``rnn_class``: RNN class (``nn.RNN``, ``nn.LSTM``, or ``nn.GRU``)
- ``dropout_rate``: Dropout rate between RNN layers

Dilated CNN Model
-----------------

The Dilated CNN model uses dilated convolutions to capture long-range dependencies
in time series data. It's particularly effective for sequences with periodic patterns.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from deep_time_series.model import DilatedCNN
    
    model = DilatedCNN(
        hidden_size=64,
        encoding_length=30,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=['feature'],
        dilation_base=2,
        kernel_size=3,
        activation=torch.nn.ELU,
        dropout_rate=0.1,
    )
    
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=model.make_chunk_specs()
    )
    dataloader = DataLoader(dataset, batch_size=32)
    
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=dataloader)

How Dilated CNN Works
~~~~~~~~~~~~~~~~~~~~~

The model automatically calculates the number of layers needed based on:
- ``encoding_length``: The input sequence length
- ``dilation_base``: Base for exponential dilation (e.g., 2 means dilations: 1, 2, 4, 8, ...)
- ``kernel_size``: Size of convolutional kernel

The dilation increases exponentially with each layer, allowing the model to capture
dependencies at different time scales.

Parameters
~~~~~~~~~~

- ``hidden_size``: Number of convolutional filters
- ``encoding_length``: Length of encoding window
- ``decoding_length``: Length of decoding window
- ``target_names``: List of target feature names
- ``nontarget_names``: List of non-target feature names
- ``dilation_base``: Base for exponential dilation (typically 2)
- ``kernel_size``: Size of convolutional kernel (must be >= dilation_base)
- ``activation``: Activation function class (default: ``nn.ELU``)
- ``dropout_rate``: Dropout rate (default: 0.0)

Transformer Model
-----------------

The SingleShotTransformer model uses a transformer architecture with encoder-decoder structure.
It's effective for capturing complex temporal dependencies and long-range patterns.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from deep_time_series.model import SingleShotTransformer
    
    model = SingleShotTransformer(
        encoding_length=30,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=['feature'],
        d_model=128,
        n_heads=8,
        n_layers=4,
        dim_feedforward=512,
        dropout_rate=0.1,
    )
    
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=model.make_chunk_specs()
    )
    dataloader = DataLoader(dataset, batch_size=32)
    
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=dataloader)

Transformer Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

The transformer uses:
- **Encoder**: Processes the encoding window with self-attention
- **Decoder**: Generates predictions using cross-attention to encoder outputs
- **Positional Encoding**: Adds positional information to inputs
- **Causal Masking**: Prevents decoder from seeing future information during training

Parameters
~~~~~~~~~~

- ``encoding_length``: Length of encoding window
- ``decoding_length``: Length of decoding window
- ``target_names``: List of target feature names
- ``nontarget_names``: List of non-target feature names
- ``d_model``: Dimension of model (embedding size)
- ``n_heads``: Number of attention heads
- ``n_layers``: Number of encoder/decoder layers
- ``dim_feedforward``: Dimension of feedforward network (default: 4 * d_model)
- ``dropout_rate``: Dropout rate (default: 0.0)

Model Comparison
----------------

Choosing the Right Model
~~~~~~~~~~~~~~~~~~~~~~~~~

**MLP**
    - Simple and fast
    - Good for short sequences and simple patterns
    - No explicit temporal modeling

**RNN (LSTM/GRU)**
    - Good for sequential dependencies
    - Can handle variable-length sequences
    - May struggle with very long sequences

**Dilated CNN**
    - Efficient for long sequences
    - Good for periodic patterns
    - Parallel processing (faster than RNN)

**Transformer**
    - Best for complex patterns and long-range dependencies
    - Most flexible but computationally expensive
    - Requires more data to train effectively

Feature Support
~~~~~~~~~~~~~~~

All models support:

- **Target features**: Variables to predict
- **Non-target features**: Additional features known at prediction time
- **Deterministic forecasting**: Point predictions
- **Probabilistic forecasting**: Distribution predictions (with DistributionHead)

Example: Using Non-Target Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All models can use non-target features:

.. code-block:: python

    # Model with non-target features
    model = MLP(
        hidden_size=64,
        encoding_length=10,
        decoding_length=5,
        target_names=['target'],
        nontarget_names=['feature1', 'feature2'],  # Multiple features
        n_hidden_layers=2,
    )

The non-target features are:
- Used during encoding (along with target features)
- Available during decoding (future values must be known)

Customizing Models
------------------

All models support custom heads, loss functions, and optimizers:

.. code-block:: python

    import torch.nn as nn
    from deep_time_series.core import Head
    
    # Custom head with L1 loss
    custom_head = Head(
        tag='targets',
        output_module=nn.Linear(64, 1),
        loss_fn=nn.L1Loss(),
        loss_weight=1.0,
    )
    
    model = MLP(
        hidden_size=64,
        encoding_length=10,
        decoding_length=5,
        target_names=['target'],
        nontarget_names=[],
        n_hidden_layers=2,
        head=custom_head,  # Use custom head
    )

For more advanced customization, see :doc:`advanced`.

