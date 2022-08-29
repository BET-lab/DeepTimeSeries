import sys
from typing import Type

from deep_time_series.data.dataset import (
    _merge_data_frames,
    TimeSeriesDataset,
)

from deep_time_series import (
    EncodingChunkSpec,
)

sys.path.append('..')

import pytest

import logging
logger = logging.getLogger('test')

import numpy as np
import pandas as pd


def test_dataset():
    df = pd.DataFrame(
        data={
            'time_index': np.arange(20),
            'a': np.arange(20)**2,
        }
    )

    merged_df = _merge_data_frames(df)

    assert '__time_index' in merged_df.columns
    assert '__time_series_id' in merged_df.columns

    # spec = EncodingChunkSpec(
    #     tag='my_tag',
    #     names=['a'],
    #     range_=(1, 3),
    #     dtype=np.float32
    # )

    # dataset = TimeSeriesDataset(
    #     df=df,
    # )