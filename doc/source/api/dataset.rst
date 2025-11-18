Dataset
=======

The dataset module provides the :class:`TimeSeriesDataset` class for loading and processing time series data in a format suitable for PyTorch DataLoader.

TimeSeriesDataset
-----------------

A PyTorch Dataset class that handles time series data extraction using chunk specifications. It supports single or multiple DataFrames and automatically extracts chunks based on the provided specifications.

**Key Features:**

- **PyTorch Compatible**: Inherits from ``torch.utils.data.Dataset``, works seamlessly with ``DataLoader``
- **Multiple DataFrames**: Supports both single DataFrame and list of DataFrames (useful for multiple time series)
- **Automatic Chunk Extraction**: Uses ``ChunkExtractor`` internally to extract chunks based on specifications
- **Time Index Tracking**: Optionally returns time indices for each chunk

**Typical Usage:**

1. Create a model (e.g., ``MLP``) which generates chunk specifications via ``make_chunk_specs()``
2. Create a ``TimeSeriesDataset`` with your data and the chunk specifications
3. Use with PyTorch ``DataLoader`` for batching

**Example:**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from torch.utils.data import DataLoader
   from deep_time_series.dataset import TimeSeriesDataset
   from deep_time_series.model import MLP

   # Create sample data
   data = pd.DataFrame({
       'temperature': np.sin(np.arange(100)),
       'humidity': np.cos(np.arange(100)),
   })

   # Create model
   model = MLP(
       hidden_size=64,
       encoding_length=10,
       decoding_length=5,
       target_names=['temperature'],
       nontarget_names=['humidity'],
       n_hidden_layers=2,
   )

   # Get chunk specifications from model
   chunk_specs = model.make_chunk_specs()

   # Create dataset
   dataset = TimeSeriesDataset(
       data_frames=data,
       chunk_specs=chunk_specs,
       return_time_index=True,  # Include time indices in output
   )

   # Use with DataLoader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # Iterate over batches
   for batch in dataloader:
       # batch is a dictionary with keys like:
       # - 'encoding.targets': tensor of shape (batch_size, 10, 1)
       # - 'label.targets': tensor of shape (batch_size, 5, 1)
       # - 'encoding.targets.time_index': tensor of shape (batch_size, 10)
       # - 'label.targets.time_index': tensor of shape (batch_size, 5)
       pass

**Multiple DataFrames:**

When working with multiple time series (e.g., different sensors or locations), pass a list of DataFrames:

.. code-block:: python

   # Multiple time series
   data1 = pd.DataFrame({'temperature': np.sin(np.arange(100))})
   data2 = pd.DataFrame({'temperature': np.cos(np.arange(100))})

   dataset = TimeSeriesDataset(
       data_frames=[data1, data2],
       chunk_specs=chunk_specs,
   )

   # The dataset will concatenate all series and track which series each sample comes from

**Dataset Length:**

The length of the dataset is determined by:
- The length of each DataFrame
- The chunk specifications (specifically, the minimum time index)

The formula is approximately: ``len(df) - chunk_length + 1`` for each DataFrame.

**Visualization:**

You can visualize the chunk specifications using the ``plot_chunks()`` method:

.. code-block:: python

   dataset.plot_chunks()  # Shows a horizontal bar chart of chunk windows

.. automodule:: deep_time_series.dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: deep_time_series.dataset.TimeSeriesDataset
   :members:
   :undoc-members:
   :show-inheritance:
