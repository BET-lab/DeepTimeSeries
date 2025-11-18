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

- **``encode(inputs)``**: Process the encoding window and return encoder outputs. This method receives a dictionary with keys like ``'encoding.targets'`` and ``'encoding.nontargets'`` containing tensors of shape ``(batch_size, encoding_length, n_features)``. Should return a dictionary of encoder outputs that will be passed to the decoder.

- **``decode_eval(inputs)``**: Generate predictions during evaluation/inference. Receives a dictionary combining the original inputs and encoder outputs. Should return a dictionary with head tags as keys (e.g., ``'head.targets'``) and prediction tensors as values.

- **``decode_train(inputs)``**: (Optional) Generate predictions during training. Defaults to ``decode_eval()`` if not overridden. Can be used to implement teacher forcing or other training-specific behaviors.

- **``make_chunk_specs()``**: Generate chunk specifications based on model parameters. Should return a list of ``BaseChunkSpec`` instances that define the input/output structure. This is used by ``TimeSeriesDataset`` to extract the correct data windows.

**Key Methods:**

- **``forward(inputs)``**: Main forward pass that combines encoding and decoding. Automatically merges encoder outputs with inputs for the decoder. Returns the final output dictionary.

- **``calculate_loss(outputs, batch)``**: Computes the total loss by aggregating losses from all heads with their respective weights. Returns a scalar tensor.

- **``training_step(batch, batch_idx)``**: PyTorch Lightning training step. Computes forward pass, loss, and metrics. Automatically called during training.

- **``validation_step(batch, batch_idx)``**: PyTorch Lightning validation step. Computes forward pass and loss, updates metrics. Metrics are computed at epoch end.

- **``test_step(batch, batch_idx)``**: PyTorch Lightning test step. Similar to validation step but for testing.

- **``forward_metrics(outputs, batch, stage)``**: Computes metrics for the current batch. Returns a dictionary of metric values.

- **``update_metrics(outputs, batch, stage)``**: Updates metric states without computing values. Used during validation/test.

- **``compute_metrics(stage)``**: Computes final metric values from accumulated states. Called at epoch end.

- **``reset_metrics(stage)``**: Resets metric states. Called at the start of each epoch.

**Properties:**

- **``encoding_length``** (int): Length of the input window for encoding. Must be a positive integer.

- **``decoding_length``** (int): Length of the prediction window. Must be a positive integer.

- **``head``** (BaseHead): Single output head (convenience property when using a single head). Raises an error if multiple heads are defined.

- **``heads``** (list[BaseHead]): List of output heads. Supports multi-head models where different heads can predict different targets or use different loss functions.

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

- **``forward(inputs)``**: Process inputs and produce predictions. This method is called during autoregressive decoding for each time step. Should return a tensor representing the prediction for the current step. The head accumulates these predictions internally.

- **``get_outputs()``**: Return all accumulated outputs as a dictionary. After autoregressive decoding completes, this method concatenates all predictions from ``forward()`` calls. Returns a dictionary with the head tag as key and a tensor of shape ``(batch_size, decoding_length, n_features)`` as value.

- **``reset()``**: Reset internal state (called at the start of each forward pass). Clears any accumulated predictions or internal state. Must be called before starting a new autoregressive decoding sequence.

- **``calculate_loss(outputs, batch)``**: Calculate loss between outputs and labels. Receives the output dictionary from ``get_outputs()`` and the batch dictionary containing labels. Should return a scalar tensor representing the loss.

**Key Properties:**

- **``tag``** (str): Unique identifier for the head. Automatically prefixed with ``'head.'`` (e.g., ``'head.targets'``).

- **``loss_weight``** (float): Weight for this head's loss in the total loss calculation. Default is 1.0. Used when multiple heads are present to balance their contributions.

- **``metrics``** (MetricModule | None): Optional metrics to track. If set, metrics are automatically updated during training/validation/test steps.

- **``label_tag``** (str): Corresponding label tag. Automatically derived from the head tag (e.g., ``'head.targets'`` → ``'label.targets'``).

- **``has_metrics``** (bool): Whether metrics are configured for this head.

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

**Purpose:**

Use ``Head`` when you want to produce single point predictions (deterministic forecasting). This is the most common type of head for time series forecasting.

**Key Features:**

- Accumulates predictions during autoregressive decoding step by step
- Supports custom loss functions (default: ``nn.MSELoss()``)
- Optional metric tracking via TorchMetrics
- Configurable loss weight for multi-head models

**Initialization Parameters:**

- ``tag`` (str): Unique identifier for the head (e.g., ``'targets'``). Will be prefixed with ``'head.'`` automatically.

- ``output_module`` (nn.Module): PyTorch module that transforms hidden states to predictions. Typically a ``nn.Linear`` layer mapping from hidden size to number of output features.

- ``loss_fn`` (Callable): Loss function that takes predictions and targets as arguments. Should return a scalar tensor. Default is ``nn.MSELoss()``.

- ``loss_weight`` (float): Weight for this head's loss in multi-head models. Default is 1.0.

- ``metrics`` (Metric | list[Metric] | dict[str, Metric] | None): Optional metrics to track. Can be a single metric, list of metrics, or dictionary of metrics. Examples: ``MeanAbsoluteError()``, ``MeanSquaredError()``.

**How It Works:**

During autoregressive decoding:

1. ``reset()`` is called to clear internal state
2. For each time step, ``forward(inputs)`` is called, which:
   - Processes inputs through ``output_module``
   - Accumulates predictions internally
   - Returns the prediction for the current step
3. After all steps, ``get_outputs()`` concatenates all predictions
4. ``calculate_loss()`` computes the loss using the accumulated outputs

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

**Purpose:**

Use ``DistributionHead`` when you want to model uncertainty in predictions. Instead of producing a single value, it produces a probability distribution from which you can sample or compute statistics.

**Key Features:**

- Supports any PyTorch distribution (e.g., ``torch.distributions.Normal``, ``torch.distributions.StudentT``)
- Automatically creates linear layers for distribution parameters
- Applies appropriate transformations to ensure valid parameter values (e.g., softplus for scale parameters)
- Uses negative log-likelihood as the loss function

**Initialization Parameters:**

- ``tag`` (str): Unique identifier for the head (e.g., ``'targets'``). Will be prefixed with ``'head.'`` automatically.

- ``distribution`` (torch.distributions.Distribution): The distribution class to use (not an instance). Common choices include ``dist.Normal``, ``dist.StudentT``, ``dist.Gamma``.

- ``in_features`` (int): Number of input features (hidden size from the model).

- ``out_features`` (int): Number of output features (number of targets to predict).

- ``loss_weight`` (float): Weight for this head's loss. Default is 1.0.

- ``metrics`` (Metric | list[Metric] | dict[str, Metric] | None): Optional metrics to track.

**How It Works:**

1. For each distribution parameter (e.g., ``loc``, ``scale`` for Normal), a linear layer is created
2. During forward pass, inputs are passed through these linear layers
3. Transformations are applied to ensure valid parameter values (e.g., ``transform_to(constraint)``)
4. A distribution instance is created with these parameters
5. A sample is drawn from the distribution as the prediction
6. Both the sample and the parameters are stored in outputs
7. Loss is computed as negative log-likelihood: ``-log_prob(targets)``

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

``MetricModule`` wraps TorchMetrics to provide stage-aware metric tracking. It automatically creates separate metric instances for training, validation, and testing phases, ensuring metrics are properly isolated and logged.

**Key Features:**

- Automatic stage separation (train/val/test) with separate metric instances
- Prefix management for logging (e.g., ``train/targets.mae``, ``val/targets.mae``)
- Tag-based organization (automatically links head tags to label tags)

**Initialization Parameters:**

- ``tag`` (str): The head tag (e.g., ``'targets'`` or ``'head.targets'``). Used to generate logging prefixes.

- ``metrics`` (Metric | list[Metric] | dict[str, Metric]): The metrics to track. Can be a single metric, list, or dictionary. Examples: ``MeanAbsoluteError()``, ``[MeanAbsoluteError(), MeanSquaredError()]``.

**Methods:**

- **``forward(outputs, batch, stage)``**: Computes metrics for the current batch. Returns a dictionary of metric values with stage prefix.

- **``update(outputs, batch, stage)``**: Updates metric states without computing values. Used during validation/test to accumulate statistics.

- **``compute(stage)``**: Computes final metric values from accumulated states. Called at epoch end.

- **``reset(stage)``**: Resets metric states for the given stage. Called at the start of each epoch.

**Internal Properties:**

- ``head_tag`` (str): The full head tag (e.g., ``'head.targets'``)
- ``label_tag`` (str): The corresponding label tag (e.g., ``'label.targets'``)

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
