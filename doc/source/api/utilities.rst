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

**When to Use:**

- Validating that multiple sets have no common elements
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

**Note:**

Typically used internally by the framework for validation purposes.

.. autofunction:: deep_time_series.util.logical_and_for_set_list

logical_or_for_set_list
~~~~~~~~~~~~~~~~~~~~~~~

Compute the union of a list of sets.

**Purpose:**

Finds all unique elements across all sets in a list. Used internally for combining sets.

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

**Note:**

Typically used internally by the framework.

.. autofunction:: deep_time_series.util.logical_or_for_set_list

merge_dicts
~~~~~~~~~~~~

Merge multiple dictionaries into a single dictionary. Raises an error if keys are duplicated.

**Purpose:**

Combines multiple dictionaries into one, ensuring no key conflicts. Used extensively in the framework to merge encoder outputs with decoder inputs.

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

.. autofunction:: deep_time_series.util.merge_dicts

merge_data_frames
~~~~~~~~~~~~~~~~~~

Merge multiple pandas DataFrames by concatenating them and adding time_index and time_series_id columns.

**Purpose:**

Combines multiple time series DataFrames into a single DataFrame while preserving information about which series each row belongs to.

**When to Use:**

- Combining multiple time series for analysis
- Preparing data from multiple sources
- Creating a unified dataset from separate series

**Key Features:**

- **Time Index Preservation**: Adds original index as 'time_index' column
- **Series Identification**: Adds 'time_series_id' to track source DataFrame
- **Deep Copy**: Creates copies to avoid modifying original DataFrames

**Example:**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from deep_time_series.util import merge_data_frames

   # Multiple time series
   df1 = pd.DataFrame({'temperature': np.sin(np.arange(100))})
   df2 = pd.DataFrame({'temperature': np.cos(np.arange(100))})
   df3 = pd.DataFrame({'temperature': np.random.randn(100)})

   # Merge with tracking
   merged = merge_data_frames([df1, df2, df3])
   # Result has columns: ['temperature', 'time_index', 'time_series_id']
   # time_series_id: 0 for df1, 1 for df2, 2 for df3

**Output Format:**

The merged DataFrame includes:
- All original columns from input DataFrames
- ``time_index``: Original index values
- ``time_series_id``: Integer ID (0, 1, 2, ...) indicating source DataFrame

**Use Cases:**

- Combining data from multiple sensors/locations
- Merging training and validation sets for preprocessing
- Creating unified datasets for analysis

.. autofunction:: deep_time_series.util.merge_data_frames

Plotting
--------

The plotting module provides visualization utilities for time series data.

plot_chunks
~~~~~~~~~~~

Visualize chunk specifications as horizontal bars showing the time windows for encoding, decoding, and labels.

**Purpose:**

Creates a visual representation of chunk specifications, making it easy to understand the temporal structure of your model's input/output windows.

**When to Use:**

- Understanding model architecture
- Debugging chunk specifications
- Visualizing data windows
- Documentation and presentations

**Output:**

Creates a horizontal bar chart where:
- Each bar represents a chunk specification
- Bar position shows the time range
- Bar width shows the window length
- Labels show the chunk tag

**Example:**

.. code-block:: python

   import matplotlib.pyplot as plt
   from deep_time_series.plotting import plot_chunks
   from deep_time_series.chunk import EncodingChunkSpec, LabelChunkSpec
   import numpy as np

   # Create chunk specifications
   chunk_specs = [
       EncodingChunkSpec('targets', ['temp'], (0, 10), np.float32),
       LabelChunkSpec('targets', ['temp'], (10, 15), np.float32),
   ]

   # Visualize
   plot_chunks(chunk_specs)
   plt.show()

**Integration with TimeSeriesDataset:**

The ``TimeSeriesDataset`` class provides a convenience method:

.. code-block:: python

   from deep_time_series.dataset import TimeSeriesDataset

   dataset = TimeSeriesDataset(data_frames=data, chunk_specs=chunk_specs)
   dataset.plot_chunks()  # Visualize the chunks used by this dataset

**Visualization Details:**

- Uses matplotlib's ``barh()`` for horizontal bars
- Alpha transparency for overlapping bars
- Annotations show chunk tags
- Y-axis shows different chunks

**Note:**

Requires matplotlib to be installed. The function modifies the current matplotlib figure/axes.

.. automodule:: deep_time_series.plotting
   :members:
   :undoc-members:

.. autofunction:: deep_time_series.plotting.plot_chunks
