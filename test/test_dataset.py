import sys

from deep_time_series import (
    ColumnTransformer,
    EncodingChunkSpec,
    LabelChunkSpec,
)
from deep_time_series.dataset import TimeSeriesDataset

sys.path.append('..')

import logging

import pytest

logger = logging.getLogger('test')

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def test_dataset():
    df = pd.DataFrame(
        data={
            'a': np.arange(20) ** 2,
            'b': -np.arange(20),
        }
    )

    logger.debug(df)

    spec = EncodingChunkSpec(
        tag='my_tag', names=['a', 'b'], range_=(-1, 3), dtype=np.float32
    )

    column_transformer = ColumnTransformer(
        transformer_tuples=[
            (
                FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x),
                ['a', 'b'],
            ),
        ]
    )

    df = column_transformer.fit_transform(df)
    logger.debug(df)

    dataset = TimeSeriesDataset(
        data_frames=df,
        chunk_specs=[spec],
    )

    assert np.allclose(dataset[0]['encoding.my_tag'], df.values[:4])

    logger.debug(f'{len(df)} = len(df)')
    logger.debug(dataset.chunk_extractors[0].chunk_length)
    logger.debug(dataset.lengths)

    assert len(dataset) == 20 - 4 + 1

    specs = [
        EncodingChunkSpec(
            tag='my_tag', names=['a', 'b'], range_=(-1, 3), dtype=np.float32
        ),
        LabelChunkSpec(
            tag='my_tag', names=['a', 'b'], range_=(2, 4), dtype=np.float32
        ),
    ]

    column_transformer = ColumnTransformer(
        transformer_tuples=[
            (
                FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x),
                ['a', 'b'],
            ),
        ]
    )

    df = column_transformer.fit_transform(df)

    dataset = TimeSeriesDataset(
        data_frames=df,
        chunk_specs=specs,
    )

    logger.debug(f'{len(df)} = len(df)')
    logger.debug(dataset.chunk_extractors[0].chunk_length)
    logger.debug(dataset.lengths)

    assert len(dataset) == 16

    # One time step shifted to the right due to the offset of encoding.my_tag.
    assert np.allclose(dataset[0]['label.my_tag'], df.values[3:5])

    dataset = TimeSeriesDataset(
        data_frames=df,
        chunk_specs=specs,
        return_time_index=True,
    )

    assert np.allclose(
        dataset[0]['encoding.my_tag.time_index'], np.arange(0, 4)
    )

    assert np.allclose(dataset[0]['label.my_tag.time_index'], np.arange(3, 5))

    specs = [
        EncodingChunkSpec(
            tag='my_tag', names=['a', 'b'], range_=(-1, 3), dtype=np.float32
        ),
    ] * 2

    # Tag duplication.
    with pytest.raises(ValueError) as e:
        dataset = TimeSeriesDataset(
            data_frames=df,
            chunk_specs=specs,
        )

    logger.info(str(e.value))
