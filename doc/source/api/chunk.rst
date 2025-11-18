Chunk Specification
===================

The chunk module provides classes for defining and extracting chunks from time series data. Chunks are used to specify the input windows for encoding, decoding, and the target windows for prediction.

.. automodule:: deep_time_series.chunk
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   deep_time_series.chunk.BaseChunkSpec
   deep_time_series.chunk.EncodingChunkSpec
   deep_time_series.chunk.DecodingChunkSpec
   deep_time_series.chunk.LabelChunkSpec
   deep_time_series.chunk.ChunkExtractor
   deep_time_series.chunk.ChunkInverter

BaseChunkSpec
~~~~~~~~~~~~~

Base class for all chunk specifications. Defines the structure for specifying time windows and feature names.

.. autoclass:: deep_time_series.chunk.BaseChunkSpec
   :members:
   :undoc-members:
   :show-inheritance:

EncodingChunkSpec
~~~~~~~~~~~~~~~~~

Specification for the encoding window, which defines the input features for the encoder.

.. autoclass:: deep_time_series.chunk.EncodingChunkSpec
   :members:
   :undoc-members:
   :show-inheritance:

DecodingChunkSpec
~~~~~~~~~~~~~~~~~

Specification for the decoding window, which defines the input features for the decoder during autoregressive prediction.

.. autoclass:: deep_time_series.chunk.DecodingChunkSpec
   :members:
   :undoc-members:
   :show-inheritance:

LabelChunkSpec
~~~~~~~~~~~~~~

Specification for the label window, which defines the target values for prediction.

.. autoclass:: deep_time_series.chunk.LabelChunkSpec
   :members:
   :undoc-members:
   :show-inheritance:

ChunkExtractor
~~~~~~~~~~~~~~

Utility class for extracting chunks from pandas DataFrames based on chunk specifications.

.. autoclass:: deep_time_series.chunk.ChunkExtractor
   :members:
   :undoc-members:
   :show-inheritance:

ChunkInverter
~~~~~~~~~~~~~

Utility class for converting model outputs (tensors) back to pandas DataFrames.

.. autoclass:: deep_time_series.chunk.ChunkInverter
   :members:
   :undoc-members:
   :show-inheritance:

