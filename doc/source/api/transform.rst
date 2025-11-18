Data Transformation
====================

The transform module provides data preprocessing utilities for time series data. The main class is ColumnTransformer, which allows applying different transformers to different columns of a DataFrame.

ColumnTransformer
------------------

A transformer that applies sklearn-style transformers to specific columns of a pandas DataFrame. It supports both dictionary and tuple-based transformer specifications.

.. automodule:: deep_time_series.transform
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.transform.ColumnTransformer

.. autoclass:: deep_time_series.transform.ColumnTransformer
   :members:
   :undoc-members:
   :show-inheritance:

