Utilities
==========

Utility Functions
-----------------

The util module provides helper functions for data manipulation and dictionary operations. These are internal utilities used by the framework but may also be useful for custom implementations.

**Purpose:**

These functions handle common operations needed when working with time series data and model outputs:
- Set operations for validation
- Dictionary merging for combining model outputs
- DataFrame merging for handling multiple time series

logical_and_for_set_list
~~~~~~~~~~~~~~~~~~~~~~~~

Compute the intersection of a list of sets.

**Purpose:**

Finds elements that are common to all sets in a list. Used internally for validation (e.g., checking for duplicate keys in dictionaries).

**Parameters:**

- ``set_list`` (list[set]): List of sets to compute the intersection of. Must contain at least one set.

**Returns:**

- ``set``: A set containing elements that appear in all input sets. If any set is empty or there are no common elements, returns an empty set.

**When to Use:**

- Validating that multiple sets have no common elements (check if result is empty)
- Finding common elements across multiple sets
- Internal validation in framework code

**Example:**

.. code-block:: python

   from deep_time_series.util import logical_and_for_set_list

   sets = [
       {'a', 'b', 'c'},
       {'b', 'c', 'd'},
       {'c', 'd', 'e'},
   ]
   common = logical_and_for_set_list(sets)  # {'c'}
   
   # Check for duplicates (common use case)
   if logical_and_for_set_list([set(d1.keys()), set(d2.keys())]):
       raise ValueError("Duplicate keys found!")

**Note:**

Typically used internally by the framework for validation purposes. The function computes the intersection sequentially: ``set1 & set2 & set3 & ...``.

.. autofunction:: deep_time_series.util.logical_and_for_set_list

logical_or_for_set_list
~~~~~~~~~~~~~~~~~~~~~~~

Compute the union of a list of sets.

**Purpose:**

Finds all unique elements across all sets in a list. Used internally for combining sets.

**Parameters:**

- ``set_list`` (list[set]): List of sets to compute the union of. Must contain at least one set.

**Returns:**

- ``set``: A set containing all unique elements from all input sets.

**When to Use:**

- Combining multiple sets into one
- Finding all unique elements across sets
- Internal operations in framework code

**Example:**

.. code-block:: python

   from deep_time_series.util import logical_or_for_set_list

   sets = [
       {'a', 'b'},
       {'b', 'c'},
       {'c', 'd'},
   ]
   union = logical_or_for_set_list(sets)  # {'a', 'b', 'c', 'd'}
   
   # Get all unique keys from multiple dictionaries
   all_keys = logical_or_for_set_list([set(d.keys()) for d in dict_list])

**Note:**

Typically used internally by the framework. The function computes the union sequentially: ``set1 | set2 | set3 | ...``.

.. autofunction:: deep_time_series.util.logical_or_for_set_list

merge_dicts
~~~~~~~~~~~~

Merge multiple dictionaries into a single dictionary. Raises an error if keys are duplicated.

**Purpose:**

Combines multiple dictionaries into one, ensuring no key conflicts. Used extensively in the framework to merge encoder outputs with decoder inputs.

**Parameters:**

- ``dicts`` (list[dict]): List of dictionaries to merge. All dictionaries will be combined into one.

- ``ignore_keys`` (set | list[str] | None): Optional set or list of keys to ignore during merging. These keys will be excluded from the result even if they appear in multiple dictionaries. Default is ``None``.

**Returns:**

- ``dict``: A new dictionary containing all key-value pairs from all input dictionaries (except ignored keys). Maintains insertion order (Python 3.7+).

**Raises:**

- ``AssertionError``: If any keys overlap between dictionaries (unless they are in ``ignore_keys``).

**When to Use:**

- Combining model outputs from different stages (e.g., encoder + decoder)
- Merging multiple dictionaries without key conflicts
- Internal framework operations

**Key Features:**

- **Duplicate Detection**: Raises an assertion error if any keys overlap
- **Key Filtering**: Can ignore specific keys during merging
- **Order Preservation**: Maintains insertion order (Python 3.7+)

**Example:**

.. code-block:: python

   from deep_time_series.util import merge_dicts

   dict1 = {'a': 1, 'b': 2}
   dict2 = {'c': 3, 'd': 4}
   merged = merge_dicts([dict1, dict2])  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}

   # With ignore_keys
   dict1 = {'a': 1, 'b': 2, 'temp': 999}
   dict2 = {'c': 3}
   merged = merge_dicts([dict1, dict2], ignore_keys=['temp'])
   # {'a': 1, 'b': 2, 'c': 3}

**Use in ForecastingModule:**

The ``ForecastingModule.forward()`` method uses this to merge encoder outputs with inputs for the decoder:

.. code-block:: python

   encoder_outputs = self.encode(inputs)
   decoder_inputs = merge_dicts([inputs, encoder_outputs])
   outputs = self.decode(decoder_inputs)

**Important:**

- Keys must be unique across all dictionaries (unless in ignore_keys)
- Raises AssertionError if duplicates are found
- The function creates a new dictionary; original dictionaries are not modified

.. autofunction:: deep_time_series.util.merge_dicts

merge_data_frames
~~~~~~~~~~~~~~~~~~

Merge multiple pandas DataFrames by concatenating them and adding time_index and time_series_id columns.

**Purpose:**

Combines multiple time series DataFrames into a single DataFrame while preserving information about which series each row belongs to. This is useful when working with multiple related time series that need to be analyzed together.

**Parameters:**

- ``dfs`` (list[pd.DataFrame]): List of DataFrames to merge. Each DataFrame represents a separate time series. DataFrames should have compatible column structures (same column names and types).

**Returns:**

- ``pd.DataFrame``: A single DataFrame containing all rows from all input DataFrames, with two additional columns:
  - ``time_index``: The original index values from each DataFrame
  - ``time_series_id``: Integer identifier (0, 1, 2, ...) indicating which DataFrame each row came from

**Key Features:**

- **Time Index Preservation**: Adds original index as 'time_index' column
- **Series Identification**: Adds 'time_series_id' to track source DataFrame
- **Deep Copy**: Creates copies to avoid modifying original DataFrames
- **Index Reset**: Resets the index of the merged DataFrame (uses default integer index)

**When to Use:**

- Combining multiple time series for analysis
- Preparing data from multiple sources
- Creating a unified dataset from separate series
- Preprocessing multiple series together with ``ColumnTransformer``

**Example:**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from deep_time_series.util import merge_data_frames

   # Multiple time series from different sensors
   df1 = pd.DataFrame({
       'temperature': np.sin(np.arange(100)),
       'humidity': np.random.rand(100)
   })
   df2 = pd.DataFrame({
       'temperature': np.cos(np.arange(100)),
       'humidity': np.random.rand(100)
   })
   df3 = pd.DataFrame({
       'temperature': np.random.randn(100),
       'humidity': np.random.rand(100)
   })

   # Merge with tracking
   merged = merge_data_frames([df1, df2, df3])
   # Result has columns: ['temperature', 'humidity', 'time_index', 'time_series_id']
   # time_series_id: 0 for df1, 1 for df2, 2 for df3
   # time_index: original index values from each DataFrame

**Output Format:**

The merged DataFrame includes:
- All original columns from input DataFrames
- ``time_index``: Original index values (preserved from each source DataFrame)
- ``time_series_id``: Integer ID (0, 1, 2, ...) indicating source DataFrame

**Use Cases:**

- Combining data from multiple sensors/locations
- Merging training and validation sets for preprocessing
- Creating unified datasets for analysis
- Preparing data for models that can handle multiple time series

.. autofunction:: deep_time_series.util.merge_data_frames

Plotting
--------

The plotting module provides visualization utilities for time series data.

plot_chunks
~~~~~~~~~~~

Visualize chunk specifications as horizontal bars showing the time windows for encoding, decoding, and labels.

**Purpose:**

Creates a visual representation of chunk specifications, making it easy to understand the temporal structure of your model's input/output windows. This visualization helps debug chunk configurations and understand how data flows through the model.

**Parameters:**

- ``chunk_specs`` (list[BaseChunkSpec]): List of chunk specifications to visualize. Each chunk will be displayed as a horizontal bar.

**Returns:**

- ``None``: The function modifies the current matplotlib figure/axes in place. Use ``plt.show()`` or ``plt.savefig()`` to display or save the plot.

**When to Use:**

- Understanding model architecture
- Debugging chunk specifications
- Visualizing data windows
- Documentation and presentations
- Verifying that chunk ranges are correct

**Output:**

Creates a horizontal bar chart where:
- Each bar represents a chunk specification
- Bar position (left edge) shows the start of the time range
- Bar width shows the window length (end - start)
- Labels show the chunk tag
- Y-axis position indicates different chunks

**Example:**

.. code-block:: python

   import matplotlib.pyplot as plt
   from deep_time_series.plotting import plot_chunks
   from deep_time_series.chunk import EncodingChunkSpec, LabelChunkSpec, DecodingChunkSpec
   import numpy as np

   # Create chunk specifications
   chunk_specs = [
       EncodingChunkSpec('targets', ['temp'], (0, 10), np.float32),
       DecodingChunkSpec('nontargets', ['humidity'], (10, 15), np.float32),
       LabelChunkSpec('targets', ['temp'], (10, 15), np.float32),
   ]

   # Visualize
   plot_chunks(chunk_specs)
   plt.xlabel('Time Index')
   plt.title('Chunk Specifications')
   plt.show()

**Integration with TimeSeriesDataset:**

The ``TimeSeriesDataset`` class provides a convenience method:

.. code-block:: python

   from deep_time_series.dataset import TimeSeriesDataset

   dataset = TimeSeriesDataset(data_frames=data, chunk_specs=chunk_specs)
   dataset.plot_chunks()  # Visualize the chunks used by this dataset
   plt.show()

**Visualization Details:**

- Uses matplotlib's ``barh()`` for horizontal bars
- Alpha transparency (0.8) for overlapping bars
- Annotations show chunk tags at the left edge of each bar
- Y-axis shows different chunks (numbered from 1)
- X-axis shows time indices

**Note:**

- Requires matplotlib to be installed
- The function modifies the current matplotlib figure/axes
- You may want to add labels and title using ``plt.xlabel()``, ``plt.ylabel()``, ``plt.title()``
- Call ``plt.show()`` or ``plt.savefig()`` after calling this function to display or save the plot

.. automodule:: deep_time_series.plotting
   :members:
   :undoc-members:

.. autofunction:: deep_time_series.plotting.plot_chunks
