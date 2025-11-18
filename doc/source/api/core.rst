Core Modules
============

The core module provides the base classes for building forecasting models. These classes form the foundation of the DeepTimeSeries framework and define the interface for all forecasting modules.

ForecastingModule
-----------------

The base class for all forecasting models. It provides automatic training/validation/test step implementations, metric tracking, and loss calculation.

.. automodule:: deep_time_series.core
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.core.ForecastingModule
   deep_time_series.core.BaseHead
   deep_time_series.core.Head
   deep_time_series.core.DistributionHead
   deep_time_series.core.MetricModule

ForecastingModule
~~~~~~~~~~~~~~~~~

.. autoclass:: deep_time_series.core.ForecastingModule
   :members:
   :undoc-members:
   :show-inheritance:

BaseHead
~~~~~~~~

Base class for all head modules. Heads are responsible for producing model outputs and calculating losses.

.. autoclass:: deep_time_series.core.BaseHead
   :members:
   :undoc-members:
   :show-inheritance:

Head
~~~~

Deterministic head for producing point predictions.

.. autoclass:: deep_time_series.core.Head
   :members:
   :undoc-members:
   :show-inheritance:

DistributionHead
~~~~~~~~~~~~~~~~

Probabilistic head for producing distribution-based predictions.

.. autoclass:: deep_time_series.core.DistributionHead
   :members:
   :undoc-members:
   :show-inheritance:

MetricModule
~~~~~~~~~~~~

Module for tracking and computing metrics during training, validation, and testing.

.. autoclass:: deep_time_series.core.MetricModule
   :members:
   :undoc-members:
   :show-inheritance:

