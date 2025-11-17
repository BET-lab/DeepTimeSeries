Advanced Topics
================

This tutorial covers advanced features of DeepTimeSeries, including probabilistic forecasting,
custom models, multi-head models, and other advanced use cases.

Probabilistic Forecasting
------------------------

DeepTimeSeries supports probabilistic forecasting through ``DistributionHead``,
which allows models to predict probability distributions instead of point values.

Why Probabilistic Forecasting?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Probabilistic forecasting provides:
- **Uncertainty quantification**: Understand prediction confidence
- **Risk assessment**: Evaluate worst-case scenarios
- **Better decision making**: Make informed decisions under uncertainty

Using DistributionHead
~~~~~~~~~~~~~~~~~~~~~~~

To use probabilistic forecasting, create a ``DistributionHead`` with a PyTorch distribution:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.distributions as dist
    from deep_time_series.core import DistributionHead
    from deep_time_series.model import MLP
    
    # Create DistributionHead for Normal distribution
    prob_head = DistributionHead(
        tag='targets',
        distribution=dist.Normal,
        in_features=64,  # Output size from model body
        out_features=1,  # Number of target features
    )
    
    # Use with MLP model
    model = MLP(
        hidden_size=64,
        encoding_length=20,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=[],
        n_hidden_layers=2,
        head=prob_head,  # Use probabilistic head
    )

Supported Distributions
~~~~~~~~~~~~~~~~~~~~~~~

You can use any PyTorch distribution that supports the required operations:

.. code-block:: python

    # Normal distribution (mean and std)
    dist.Normal
    
    # Laplace distribution (loc and scale)
    dist.Laplace
    
    # Student's t-distribution (df, loc, scale)
    dist.StudentT
    
    # Log-normal distribution (loc and scale)
    dist.LogNormal
    
    # And many more...

Example: Normal Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example using Normal distribution:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    import torch.distributions as dist
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    
    import deep_time_series as dts
    from deep_time_series.core import DistributionHead
    from deep_time_series.model import MLP
    from deep_time_series.chunk import ChunkInverter
    
    # Prepare data
    data = pd.DataFrame({
        'target': np.sin(np.arange(100)) + 0.1 * np.random.randn(100)
    })
    
    transformer = dts.ColumnTransformer(
        transformer_tuples=[(StandardScaler(), ['target'])]
    )
    data = transformer.fit_transform(data)
    
    # Create probabilistic head
    prob_head = DistributionHead(
        tag='targets',
        distribution=dist.Normal,
        in_features=64,
        out_features=1,
    )
    
    # Create model
    model = MLP(
        hidden_size=64,
        encoding_length=20,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=[],
        n_hidden_layers=2,
        head=prob_head,
    )
    
    # Train
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=model.make_chunk_specs()
    )
    dataloader = DataLoader(dataset, batch_size=32)
    
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=dataloader)
    
    # Predict
    model.eval()
    batch = next(iter(dataloader))
    with torch.no_grad():
        outputs = model(batch)
    
    # Extract distribution parameters
    mean = outputs['head.targets.loc']  # Mean predictions
    std = outputs['head.targets.scale']  # Standard deviation
    
    # Sample from distribution
    samples = torch.distributions.Normal(mean, std).sample()

Accessing Distribution Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``DistributionHead``, the outputs dictionary contains:
- ``head.{tag}``: Sampled values
- ``head.{tag}.{param}``: Distribution parameters (e.g., ``head.targets.loc``, ``head.targets.scale``)

Multi-Head Models
-----------------

Multi-head models allow predicting multiple targets or using different loss functions
for different outputs.

Use Cases
~~~~~~~~~

- **Multiple targets**: Predict several related variables simultaneously
- **Different loss functions**: Use different losses for different outputs
- **Auxiliary tasks**: Predict auxiliary information alongside main target

Creating Multi-Head Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch.nn as nn
    from deep_time_series.core import Head
    from deep_time_series.model import MLP
    
    # Create multiple heads
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
            loss_weight=0.5,  # Lower weight for this head
        ),
    ]
    
    # Create model
    model = MLP(
        hidden_size=64,
        encoding_length=20,
        decoding_length=10,
        target_names=['target1', 'target2'],  # Multiple targets
        nontarget_names=[],
        n_hidden_layers=2,
    )
    
    # Set multiple heads
    model.heads = heads

Chunk Specifications for Multi-Head
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using multiple heads, you need to define chunk specifications for each target:

.. code-block:: python

    from deep_time_series.chunk import EncodingChunkSpec, LabelChunkSpec
    
    # Model will automatically create specs for all targets
    chunk_specs = model.make_chunk_specs()
    
    # Or create manually
    chunk_specs = [
        EncodingChunkSpec(tag='target1', names=['target1'], range_=(0, 20), dtype=np.float32),
        LabelChunkSpec(tag='target1', names=['target1'], range_=(20, 30), dtype=np.float32),
        EncodingChunkSpec(tag='target2', names=['target2'], range_=(0, 20), dtype=np.float32),
        LabelChunkSpec(tag='target2', names=['target2'], range_=(20, 30), dtype=np.float32),
    ]

Loss Weighting
~~~~~~~~~~~~~~

Each head can have a different loss weight:

.. code-block:: python

    heads = [
        Head(..., loss_weight=1.0),   # Main target
        Head(..., loss_weight=0.5),    # Secondary target
        Head(..., loss_weight=0.1),    # Auxiliary task
    ]

The total loss is: ``loss = w1*loss1 + w2*loss2 + w3*loss3``

Custom Models
-------------

You can create custom models by inheriting from ``ForecastingModule``.

Basic Custom Model
~~~~~~~~~~~~~~~~~~

Here's a minimal example:

.. code-block:: python

    import numpy as np
    import torch
    import torch.nn as nn
    from deep_time_series.chunk import EncodingChunkSpec, LabelChunkSpec
    from deep_time_series.core import ForecastingModule, Head
    
    class SimpleCustomModel(ForecastingModule):
        def __init__(
            self,
            hidden_size,
            encoding_length,
            decoding_length,
            target_names,
            nontarget_names,
        ):
            super().__init__()
            
            self.encoding_length = encoding_length
            self.decoding_length = decoding_length
            
            n_outputs = len(target_names)
            n_features = len(nontarget_names) + n_outputs
            
            # Define encoder
            self.encoder = nn.Sequential(
                nn.Linear(n_features * encoding_length, hidden_size),
                nn.ReLU(),
            )
            
            # Define decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_outputs),
            )
            
            # Define head
            self.head = Head(
                tag='targets',
                output_module=nn.Linear(hidden_size, n_outputs),
                loss_fn=nn.MSELoss(),
            )
        
        def encode(self, inputs):
            # Concatenate target and non-target features
            if len(self.hparams.nontarget_names) > 0:
                x = torch.cat([
                    inputs['encoding.targets'],
                    inputs['encoding.nontargets']
                ], dim=2)
            else:
                x = inputs['encoding.targets']
            
            # Flatten and encode
            B = x.size(0)
            x_flat = x.view(B, -1)
            encoded = self.encoder(x_flat)
            
            return {'context': encoded}
        
        def decode_eval(self, inputs):
            context = inputs['context']
            
            self.head.reset()
            for i in range(self.decoding_length):
                # Generate prediction
                pred = self.head(context.unsqueeze(1))
            
            return self.head.get_outputs()
        
        def make_chunk_specs(self):
            E = self.encoding_length
            D = self.decoding_length
            
            specs = [
                EncodingChunkSpec(
                    tag='targets',
                    names=self.hparams.target_names,
                    range_=(0, E),
                    dtype=np.float32,
                ),
                LabelChunkSpec(
                    tag='targets',
                    names=self.hparams.target_names,
                    range_=(E, E + D),
                    dtype=np.float32,
                ),
            ]
            
            if len(self.hparams.nontarget_names) > 0:
                specs.extend([
                    EncodingChunkSpec(
                        tag='nontargets',
                        names=self.hparams.nontarget_names,
                        range_=(1, E + 1),
                        dtype=np.float32,
                    ),
                ])
            
            return specs

Using Custom Models
~~~~~~~~~~~~~~~~~~~

Use your custom model just like built-in models:

.. code-block:: python

    model = SimpleCustomModel(
        hidden_size=64,
        encoding_length=20,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=['feature'],
    )
    
    dataset = dts.TimeSeriesDataset(
        data_frames=data,
        chunk_specs=model.make_chunk_specs()
    )
    
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=dataloader)

Custom Optimizers
-----------------

All models support custom optimizers:

.. code-block:: python

    import torch.optim as optim
    
    model = MLP(
        ...,
        optimizer=optim.SGD,
        optimizer_options={'momentum': 0.9, 'weight_decay': 1e-4},
        lr=0.01,
    )

Or override ``configure_optimizers()``:

.. code-block:: python

    class CustomModel(ForecastingModule):
        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
            }

Custom Metrics
--------------

Add custom metrics to track model performance:

.. code-block:: python

    from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
    
    model = MLP(
        ...,
        metrics=[
            MeanAbsoluteError(),
            MeanSquaredError(),
            R2Score(),
        ],
    )

Metrics are automatically logged during training and validation.

Working with Multiple Time Series
---------------------------------

DeepTimeSeries supports multiple time series in a single dataset:

.. code-block:: python

    # Create multiple time series
    series1 = pd.DataFrame({'target': np.sin(np.arange(100))})
    series2 = pd.DataFrame({'target': np.cos(np.arange(50))})
    series3 = pd.DataFrame({'target': np.random.randn(75)})
    
    # Pass as list
    dataset = dts.TimeSeriesDataset(
        data_frames=[series1, series2, series3],
        chunk_specs=model.make_chunk_specs()
    )
    
    # Dataset automatically handles different lengths
    print(f"Total samples: {len(dataset)}")

The dataset will:
- Handle different lengths automatically
- Create separate chunk extractors for each series
- Combine all samples for training

Advanced Data Preprocessing
---------------------------

Using Multiple Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply different transformations to different columns:

.. code-block:: python

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    transformer = dts.ColumnTransformer(
        transformer_tuples=[
            (StandardScaler(), ['target']),
            (MinMaxScaler(), ['feature1']),
            (RobustScaler(), ['feature2']),
        ]
    )

Using Transformer Dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use a dictionary:

.. code-block:: python

    transformer = dts.ColumnTransformer(
        transformer_dict={
            'target': StandardScaler(),
            'feature1': MinMaxScaler(),
            'feature2': RobustScaler(),
        }
    )

Inverse Transformation
~~~~~~~~~~~~~~~~~~~~~~

To inverse transform predictions:

.. code-block:: python

    # Get transformer for specific columns
    target_transformer = transformer.named_transformers_['target']
    
    # Inverse transform predictions
    predictions_original_scale = target_transformer.inverse_transform(
        predictions_df.values
    )

Best Practices
--------------

1. **Data Preprocessing**: Always normalize/scale your data appropriately
2. **Chunk Specifications**: Ensure chunk ranges don't overlap incorrectly
3. **Head Reset**: Remember to call ``head.reset()`` at the start of decoding
4. **Batch Size**: Adjust batch size based on sequence length and model size
5. **Learning Rate**: Use learning rate scheduling for better convergence
6. **Validation**: Always use validation data to monitor overfitting
7. **Device Management**: PyTorch Lightning handles device placement automatically

Debugging Tips
--------------

1. **Check Chunk Specifications**: Use ``dataset.plot_chunks()`` to visualize chunk ranges
2. **Inspect Outputs**: Print output dictionary keys to verify head tags
3. **Monitor Loss**: Check if loss decreases as expected
4. **Shape Debugging**: Print tensor shapes at each step
5. **Gradient Flow**: Use ``torch.autograd.detect_anomaly()`` to find gradient issues

Example: Complete Advanced Workflow
------------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    import torch.distributions as dist
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    from torchmetrics import MeanAbsoluteError, MeanSquaredError
    
    import deep_time_series as dts
    from deep_time_series.core import DistributionHead
    from deep_time_series.model import MLP
    from deep_time_series.chunk import ChunkInverter
    
    # 1. Prepare multiple time series
    series1 = pd.DataFrame({
        'target': np.sin(np.arange(100)) + 0.1 * np.random.randn(100),
        'feature': np.cos(np.arange(100))
    })
    series2 = pd.DataFrame({
        'target': np.cos(np.arange(80)) + 0.1 * np.random.randn(80),
        'feature': np.sin(np.arange(80))
    })
    
    # 2. Preprocess
    transformer = dts.ColumnTransformer(
        transformer_tuples=[
            (StandardScaler(), ['target', 'feature'])
        ]
    )
    series1_transformed = transformer.fit_transform(series1)
    series2_transformed = transformer.transform(series2)
    
    # 3. Create probabilistic head with metrics
    prob_head = DistributionHead(
        tag='targets',
        distribution=dist.Normal,
        in_features=64,
        out_features=1,
        metrics=[MeanAbsoluteError(), MeanSquaredError()],
    )
    
    # 4. Create model
    model = MLP(
        hidden_size=64,
        encoding_length=20,
        decoding_length=10,
        target_names=['target'],
        nontarget_names=['feature'],
        n_hidden_layers=2,
        head=prob_head,
        optimizer=torch.optim.AdamW,
        optimizer_options={'weight_decay': 1e-4},
        lr=1e-3,
    )
    
    # 5. Create dataset with multiple series
    dataset = dts.TimeSeriesDataset(
        data_frames=[series1_transformed, series2_transformed],
        chunk_specs=model.make_chunk_specs()
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 6. Train with callbacks
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val/loss', patience=5),
            pl.callbacks.ModelCheckpoint(monitor='val/loss'),
        ],
    )
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    
    # 7. Predict and extract uncertainty
    model.eval()
    batch = next(iter(dataloader))
    with torch.no_grad():
        outputs = model(batch)
    
    mean = outputs['head.targets.loc']
    std = outputs['head.targets.scale']
    
    # 8. Convert to DataFrame
    inverter = ChunkInverter(model.make_chunk_specs())
    predictions_df = inverter.invert('head.targets.loc', mean)
    
    print("Predictions with uncertainty:")
    print(predictions_df.head())

