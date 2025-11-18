Dataset
=======

The dataset module provides the :class:`TimeSeriesDataset` class for loading and processing time series data in a format suitable for PyTorch DataLoader.

TimeSeriesDataset
-----------------

A PyTorch Dataset class that handles time series data extraction using chunk specifications. It supports single or multiple DataFrames and automatically extracts chunks based on the provided specifications.

**Purpose:**

``TimeSeriesDataset`` provides a PyTorch-compatible interface for loading time series data with chunk-based extraction. It automatically handles the extraction of encoding, decoding, and label chunks from your DataFrames based on chunk specifications.

**Key Features:**

- **PyTorch Compatible**: Inherits from ``torch.utils.data.Dataset``, works seamlessly with ``DataLoader``
- **Multiple DataFrames**: Supports both single DataFrame and list of DataFrames (useful for multiple time series)
- **Automatic Chunk Extraction**: Uses ``ChunkExtractor`` internally to extract chunks based on specifications
- **Time Index Tracking**: Optionally returns time indices for each chunk

**Initialization Parameters:**

- ``data_frames`` (pd.DataFrame | list[pd.DataFrame]): Input data. Can be a single DataFrame or a list of DataFrames. If multiple DataFrames are provided, they are treated as separate time series.

- ``chunk_specs`` (list[BaseChunkSpec]): List of chunk specifications defining what data to extract. Typically obtained from ``model.make_chunk_specs()``.

- ``return_time_index`` (bool): If ``True``, includes time index arrays in the output dictionary. Default is ``True``. Time indices are useful for tracking which time steps correspond to predictions.

**Properties:**

- **``data_frames``** (list[pd.DataFrame]): List of input DataFrames (always a list, even if single DataFrame was provided).

- **``chunk_specs``** (list[BaseChunkSpec]): Chunk specifications used for extraction.

- **``return_time_index``** (bool): Whether to include time indices in outputs.

- **``chunk_extractors``** (list[ChunkExtractor]): Internal list of chunk extractors, one per DataFrame.

- **``lengths``** (list[int]): List of dataset lengths for each DataFrame.

- **``min_start_time_index``** (int): Minimum valid start time index for extraction.

**Methods:**

- **``__len__()``**: Returns the total number of samples in the dataset. Calculated as the sum of lengths for all DataFrames.

- **``__getitem__(i)``**: Returns a sample at index ``i``. The index is mapped to the appropriate DataFrame and time position. Returns a dictionary with chunk tags as keys and tensors as values. If ``return_time_index=True``, also includes time index arrays.

- **``_preprocess()``**: Internal method called during initialization. Creates ``ChunkExtractor`` instances for each DataFrame and calculates dataset lengths. Automatically called by ``__init__()``.

- **``plot_chunks()``**: Visualizes the chunk specifications as a horizontal bar chart. Useful for understanding the temporal structure of your model's input/output windows. Requires matplotlib.

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
