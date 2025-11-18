Core Modules
============

The core module provides the base classes for building forecasting models. These classes form the foundation of the DeepTimeSeries framework and define the interface for all forecasting modules.

All forecasting models in DeepTimeSeries inherit from :class:`ForecastingModule`, which extends PyTorch Lightning's ``LightningModule``. This provides automatic training, validation, and testing loops, metric tracking, and loss calculation.

ForecastingModule
-----------------

The base class for all forecasting models. It provides automatic training/validation/test step implementations, metric tracking, and loss calculation.

**Key Features:**

- **PyTorch Lightning Integration**: Inherits from ``pl.LightningModule``, providing automatic training loops
- **Automatic Loss Calculation**: Aggregates losses from all heads with their respective weights
- **Metric Tracking**: Automatically updates and computes metrics for each head during training, validation, and testing
- **Encoding-Decoding Architecture**: Standardizes the encode-decode pattern for time series forecasting
- **Multi-head Support**: Can handle multiple output heads with different loss weights

**Architecture:**

The module follows an encoder-decoder pattern:

1. **Encoding**: The ``encode()`` method processes the input encoding window
2. **Decoding**: The ``decode()`` method generates predictions (can differ between training and evaluation)
3. **Forward**: Automatically combines encoding and decoding

**Required Methods to Implement:**

- ``encode()``: Process the encoding window and return encoder outputs
- ``decode_eval()``: Generate predictions during evaluation/inference
- ``decode_train()``: (Optional) Generate predictions during training (defaults to ``decode_eval()``)
- ``make_chunk_specs()``: Generate chunk specifications based on model parameters

**Properties:**

- ``encoding_length``: Length of the input window for encoding
- ``decoding_length``: Length of the prediction window
- ``head`` or ``heads``: Output head(s) for producing predictions

.. automodule:: deep_time_series.core
   :members:
   :undoc-members:
   :show-inheritance:

ForecastingModule
~~~~~~~~~~~~~~~~~

.. autoclass:: deep_time_series.core.ForecastingModule
   :members:
   :undoc-members:
   :show-inheritance:

BaseHead
~~~~~~~~

Base class for all head modules. Heads are responsible for producing model outputs and calculating losses.

**Purpose:**

Heads serve as the output layer of forecasting models. They:
- Transform encoder/decoder outputs into predictions
- Calculate loss between predictions and targets
- Track metrics (e.g., MAE, MSE, RMSE)
- Support both deterministic (point predictions) and probabilistic (distribution-based) forecasting

**Key Properties:**

- ``tag``: Unique identifier for the head (automatically prefixed with ``'head.'``)
- ``loss_weight``: Weight for this head's loss in the total loss calculation (default: 1.0)
- ``metrics``: Optional metrics to track (e.g., ``torchmetrics.MeanAbsoluteError``)
- ``label_tag``: Corresponding label tag (e.g., ``'label.targets'`` for ``'head.targets'``)

**Required Methods to Implement:**

- ``forward()``: Process inputs and produce predictions
- ``get_outputs()``: Return all accumulated outputs as a dictionary
- ``reset()``: Reset internal state (called at the start of each forward pass)
- ``calculate_loss()``: Calculate loss between outputs and labels

**Creating Custom Heads:**

To create a custom head, inherit from ``BaseHead`` and implement the required methods:

.. code-block:: python

   from deep_time_series.core import BaseHead
   import torch
   import torch.nn as nn

   class CustomHead(BaseHead):
       def __init__(self, tag, output_module, loss_fn):
           super().__init__()
           self.tag = tag
           self.output_module = output_module
           self.loss_fn = loss_fn
           self._outputs = []

       def forward(self, inputs):
           output = self.output_module(inputs)
           self._outputs.append(output)
           return output

       def get_outputs(self):
           return {self.tag: torch.cat(self._outputs, dim=1)}

       def reset(self):
           self._outputs = []

       def calculate_loss(self, outputs, batch):
           return self.loss_fn(outputs[self.tag], batch[self.label_tag])

.. autoclass:: deep_time_series.core.BaseHead
   :members:
   :undoc-members:
   :show-inheritance:

Head
~~~~

Deterministic head for producing point predictions.

**Use Case:**

Use ``Head`` when you want to produce single point predictions (e.g., the expected value). This is the most common type of head for time series forecasting.

**Features:**

- Accumulates predictions during autoregressive decoding
- Supports custom loss functions (default: ``nn.MSELoss()``)
- Optional metric tracking
- Configurable loss weight for multi-head models

**Example:**

.. code-block:: python

   from deep_time_series.core import Head
   import torch.nn as nn
   from torchmetrics import MeanAbsoluteError

   head = Head(
       tag='targets',
       output_module=nn.Linear(hidden_size, n_outputs),
       loss_fn=nn.MSELoss(),
       loss_weight=1.0,
       metrics=MeanAbsoluteError(),
   )

**Note:**

During autoregressive decoding, ``Head`` accumulates predictions step by step. Call ``reset()`` before each forward pass and ``get_outputs()`` after to retrieve all predictions.

.. autoclass:: deep_time_series.core.Head
   :members:
   :undoc-members:
   :show-inheritance:

DistributionHead
~~~~~~~~~~~~~~~~

Probabilistic head for producing distribution-based predictions.

**Use Case:**

Use ``DistributionHead`` when you want to model uncertainty in predictions. Instead of producing a single value, it produces a probability distribution (e.g., Normal, StudentT) from which you can sample or compute statistics.

**Features:**

- Supports any PyTorch distribution (e.g., ``torch.distributions.Normal``, ``torch.distributions.StudentT``)
- Automatically creates linear layers for distribution parameters
- Applies appropriate transformations to ensure valid parameter values
- Uses negative log-likelihood as the loss function

**Example:**

.. code-block:: python

   from deep_time_series.core import DistributionHead
   import torch.distributions as dist

   head = DistributionHead(
       tag='targets',
       distribution=dist.Normal,
       in_features=hidden_size,
       out_features=n_outputs,
       loss_weight=1.0,
   )

**Supported Distributions:**

Any PyTorch distribution can be used. Common choices include:

- ``dist.Normal``: For normally distributed targets
- ``dist.StudentT``: For heavy-tailed distributions
- ``dist.Gamma``: For positive-valued targets

**Output Format:**

The head produces multiple outputs:
- ``head.{tag}``: Sampled values from the distribution
- ``head.{tag}.{param}``: Distribution parameters (e.g., ``head.targets.loc``, ``head.targets.scale`` for Normal)

.. autoclass:: deep_time_series.core.DistributionHead
   :members:
   :undoc-members:
   :show-inheritance:

MetricModule
~~~~~~~~~~~~

Module for tracking and computing metrics during training, validation, and testing.

**Purpose:**

``MetricModule`` wraps TorchMetrics to provide stage-aware metric tracking. It automatically creates separate metric instances for training, validation, and testing phases.

**Features:**

- Automatic stage separation (train/val/test)
- Prefix management for logging (e.g., ``train/targets.mae``, ``val/targets.mae``)
- Tag-based organization (links head tags to label tags)

**Usage:**

Typically, you don't instantiate ``MetricModule`` directly. Instead, assign metrics to a head:

.. code-block:: python

   from deep_time_series.core import Head
   from torchmetrics import MeanAbsoluteError, MeanSquaredError

   head = Head(
       tag='targets',
       output_module=nn.Linear(hidden_size, n_outputs),
       loss_fn=nn.MSELoss(),
       metrics=[MeanAbsoluteError(), MeanSquaredError()],
   )

The ``ForecastingModule`` automatically calls the metrics during training/validation/test steps.

**Metric Tagging:**

- Head tag: ``head.{tag}`` (e.g., ``head.targets``)
- Label tag: ``label.{tag}`` (e.g., ``label.targets``)
- Logged as: ``{stage}/{tag}.{metric_name}`` (e.g., ``train/targets.mae``)

.. autoclass:: deep_time_series.core.MetricModule
   :members:
   :undoc-members:
   :show-inheritance:
