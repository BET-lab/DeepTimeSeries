Data Transformation
====================

The transform module provides data preprocessing utilities for time series data. The main class is :class:`ColumnTransformer`, which allows applying different transformers to different columns of a DataFrame.

ColumnTransformer
------------------

A transformer that applies sklearn-style transformers to specific columns of a pandas DataFrame. It supports both dictionary and tuple-based transformer specifications.

**Key Features:**

- **sklearn-Compatible Interface**: Follows the ``fit()``, ``transform()``, ``fit_transform()``, and ``inverse_transform()`` pattern
- **Column-Specific Transformers**: Apply different transformers to different columns
- **Multiple DataFrame Support**: Can fit and transform single or multiple DataFrames
- **Deep Copy Safety**: Each column gets its own transformer instance (no shared state)

**Typical Use Cases:**

- **Normalization**: Scale features to have zero mean and unit variance (``StandardScaler``)
- **Min-Max Scaling**: Scale features to a specific range (``MinMaxScaler``)
- **Robust Scaling**: Scale using median and IQR (``RobustScaler``)
- **Custom Transformers**: Apply any sklearn-compatible transformer

**Two Initialization Methods:**

1. **Dictionary Method**: Map column names directly to transformers
2. **Tuple Method**: Apply the same transformer to multiple columns (more convenient)

**Example - Dictionary Method:**

.. code-block:: python

   from deep_time_series.transform import ColumnTransformer
   from sklearn.preprocessing import StandardScaler, MinMaxScaler
   import pandas as pd
   import numpy as np

   # Create sample data
   data = pd.DataFrame({
       'temperature': np.random.randn(100) * 10 + 20,
       'humidity': np.random.rand(100) * 100,
       'pressure': np.random.randn(100) * 5 + 1013,
   })

   # Dictionary method: map each column to a transformer
   transformer = ColumnTransformer(
       transformer_dict={
           'temperature': StandardScaler(),
           'humidity': MinMaxScaler(),
           'pressure': StandardScaler(),
       }
   )

   # Fit and transform
   data_transformed = transformer.fit_transform(data)

**Example - Tuple Method (Recommended):**

.. code-block:: python

   from deep_time_series.transform import ColumnTransformer
   from sklearn.preprocessing import StandardScaler
   import pandas as pd

   # Tuple method: apply same transformer to multiple columns
   transformer = ColumnTransformer(
       transformer_tuples=[
           (StandardScaler(), ['temperature', 'pressure']),  # Scale these
           # Other columns remain unchanged
       ]
   )

   data_transformed = transformer.fit_transform(data)

**Important Notes:**

- **Deep Copying**: When using ``transformer_tuples``, each column gets a deep copy of the transformer, so they don't share state
- **Column Validation**: Only columns present in both the transformer dict and the DataFrame will be transformed
- **Preservation**: Columns not specified in the transformer will remain unchanged in the output

**Workflow:**

1. **Fit**: Compute statistics (mean, std, etc.) from training data
2. **Transform**: Apply the learned transformation to new data
3. **Inverse Transform**: Convert transformed data back to original scale (useful for predictions)

**Example - Full Workflow:**

.. code-block:: python

   from deep_time_series.transform import ColumnTransformer
   from sklearn.preprocessing import StandardScaler
   import pandas as pd

   # Training data
   train_data = pd.DataFrame({'temperature': np.random.randn(100) * 10 + 20})
   
   # Create and fit transformer
   transformer = ColumnTransformer(
       transformer_tuples=[(StandardScaler(), ['temperature'])]
   )
   train_transformed = transformer.fit_transform(train_data)

   # Test data (use same transformer, don't refit)
   test_data = pd.DataFrame({'temperature': np.random.randn(50) * 10 + 20})
   test_transformed = transformer.transform(test_data)

   # Inverse transform predictions back to original scale
   predictions_transformed = ...  # Model predictions in transformed space
   predictions_original = transformer.inverse_transform(predictions_transformed)

**Multiple DataFrames:**

The transformer can handle lists of DataFrames, which is useful when you have multiple time series:

.. code-block:: python

   data1 = pd.DataFrame({'temperature': np.random.randn(100)})
   data2 = pd.DataFrame({'temperature': np.random.randn(100)})

   transformer = ColumnTransformer(
       transformer_tuples=[(StandardScaler(), ['temperature'])]
   )

   # Fit on all data
   transformer.fit([data1, data2])

   # Transform all data
   transformed_list = transformer.transform([data1, data2])

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
