Utilities
==========

Utility Functions
-----------------

The util module provides helper functions for data manipulation and dictionary operations.

.. automodule:: deep_time_series.util
   :members:
   :undoc-members:

.. autosummary::
   :toctree: _autosummary
   :template: function.rst
   :recursive:

   deep_time_series.util.logical_and_for_set_list
   deep_time_series.util.logical_or_for_set_list
   deep_time_series.util.merge_dicts
   deep_time_series.util.merge_data_frames

logical_and_for_set_list
~~~~~~~~~~~~~~~~~~~~~~~~

Compute the intersection of a list of sets.

.. autofunction:: deep_time_series.util.logical_and_for_set_list

logical_or_for_set_list
~~~~~~~~~~~~~~~~~~~~~~~

Compute the union of a list of sets.

.. autofunction:: deep_time_series.util.logical_or_for_set_list

merge_dicts
~~~~~~~~~~~~

Merge multiple dictionaries into a single dictionary. Raises an error if keys are duplicated.

.. autofunction:: deep_time_series.util.merge_dicts

merge_data_frames
~~~~~~~~~~~~~~~~~~

Merge multiple pandas DataFrames by concatenating them and adding time_index and time_series_id columns.

.. autofunction:: deep_time_series.util.merge_data_frames

Plotting
--------

The plotting module provides visualization utilities for time series data.

.. automodule:: deep_time_series.plotting
   :members:
   :undoc-members:

.. autosummary::
   :toctree: _autosummary
   :template: function.rst
   :recursive:

   deep_time_series.plotting.plot_chunks

plot_chunks
~~~~~~~~~~~

Visualize chunk specifications as horizontal bars showing the time windows for encoding, decoding, and labels.

.. autofunction:: deep_time_series.plotting.plot_chunks

