import sys
from typing import Type

from deep_time_series.data.dataset import (
    _merge_data_frames,
    TimeSeriesDataset,
)

from deep_time_series import (
    EncodingChunkSpec,
    LabelChunkSpec,
    ColumnTransformer,
)

sys.path.append('..')

import pytest

import logging
logger = logging.getLogger('test')

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


def test_dataset():
    df = pd.DataFrame(
        data={
            'a': np.arange(20)**2,
            'b': -np.arange(20),
        }
    )

    merged_df = _merge_data_frames(df)

    assert '__time_index' in merged_df.columns
    assert '__time_series_id' in merged_df.columns

    spec = EncodingChunkSpec(
        tag='my_tag',
        names=['a', 'b'],
        range_=(-1, 3),
        dtype=np.float32
    )

    column_transformer = ColumnTransformer(
        transformer_tuples=[
            (
                FunctionTransformer(
                    func=lambda x: x,
                    inverse_func=lambda x: x
                ),
                ['a', 'b']
            ),
        ]
    )

    dataset = TimeSeriesDataset(
        data_frames=df,
        chunk_specs=[spec],
        column_transformer=column_transformer,
    )

    assert np.allclose(dataset[0]['encoding.my_tag'], df.values[:4])

    assert len(dataset) == 17

    specs = [
        EncodingChunkSpec(
            tag='my_tag',
            names=['a', 'b'],
            range_=(-1, 3),
            dtype=np.float32
        ),
        LabelChunkSpec(
            tag='my_tag',
            names=['a', 'b'],
            range_=(2, 4),
            dtype=np.float32
        ),
    ]

    column_transformer = ColumnTransformer(
        transformer_tuples=[
            (
                FunctionTransformer(
                    func=lambda x: x,
                    inverse_func=lambda x: x
                ),
                ['a', 'b']
            ),
        ]
    )

    dataset = TimeSeriesDataset(
        data_frames=df,
        chunk_specs=specs,
        column_transformer=column_transformer,
    )

    assert len(dataset) == 16

    # One time step shifted to the right due to the offset of encoding.my_tag.
    assert np.allclose(dataset[0]['label.my_tag'], df.values[3:5])

    dataset = TimeSeriesDataset(
        data_frames=df,
        chunk_specs=specs,
        column_transformer=column_transformer,
        return_time_index=True,
    )

    assert np.allclose(
        dataset[0]['encoding.my_tag.time_index'],
        np.arange(0, 4)
    )

    assert np.allclose(
        dataset[0]['label.my_tag.time_index'],
        np.arange(3, 5)
    )

    specs = [
        EncodingChunkSpec(
            tag='my_tag',
            names=['a', 'b'],
            range_=(-1, 3),
            dtype=np.float32
        ),
    ] * 2

    # Tag duplication.
    with pytest.raises(ValueError) as e:
        dataset = TimeSeriesDataset(
            data_frames=df,
            chunk_specs=specs,
            column_transformer=column_transformer,
        )

    logger.info(str(e.value))