import sys
from typing import Type

from deep_time_series.chunk import (
    EncodingChunkSpec,
    DecodingChunkSpec,
    LabelChunkSpec,
    ChunkExtractor,
)
sys.path.append('..')

import pytest

import logging
logger = logging.getLogger('test')

import numpy as np
import pandas as pd

from deep_time_series import BaseChunkSpec


def test_chunk_spec():
    chunk_spec = BaseChunkSpec()
    chunk_spec.tag = 'my_tag'

    assert chunk_spec.tag == 'my_tag'

    chunk_spec.PREFIX = 'label'
    chunk_spec.tag = 'my_tag'

    assert chunk_spec.tag == 'label.my_tag'

    with pytest.raises(TypeError):
        chunk_spec.tag = 3

    chunk_spec.names = ['a', 'b', 'c']
    assert chunk_spec.names == ['a', 'b', 'c']

    chunk_spec.names = ('a', 'b', 'c')
    assert chunk_spec.names == ['a', 'b', 'c']

    with pytest.raises(TypeError):
        chunk_spec.names = 'a'

    with pytest.raises(TypeError):
        chunk_spec.names = ['a', 3]

    chunk_spec.range = (2, 10)
    assert chunk_spec.range == (2, 10)

    with pytest.raises(TypeError):
        chunk_spec.range = 3

    with pytest.raises(TypeError):
        chunk_spec.range = (1, 'a')

    with pytest.raises(TypeError):
        chunk_spec.range = ('a', 3)

    with pytest.raises(ValueError):
        chunk_spec.range = (3, 2)

    chunk_spec.dtype = np.int32
    chunk_spec.dtype = float
    chunk_spec.dtype = int

    with pytest.raises(TypeError):
        chunk_spec.dtype = 3

    chunk_spec = EncodingChunkSpec(
        tag='my_tag',
        names=['a', 'b', 'c'],
        range_=(0, 24),
        dtype=np.float32
    )

    assert chunk_spec.tag == 'encoding.my_tag'

    chunk_spec = DecodingChunkSpec(
        tag='my_tag',
        names=['a', 'b', 'c'],
        range_=(0, 24),
        dtype=np.float32
    )

    assert chunk_spec.tag == 'decoding.my_tag'

    chunk_spec = LabelChunkSpec(
        tag='my_tag',
        names=['a', 'b', 'c'],
        range_=(0, 24),
        dtype=np.float32
    )

    assert chunk_spec.tag == 'label.my_tag'


def test_chunk_extractor():
    df = pd.DataFrame(
        data={
            '__time_index': np.arange(20),
            'a': np.arange(20)**2,
            'b': -np.arange(20),
        }
    )

    logger.info(df)

    chunk_spec = EncodingChunkSpec(
        tag='my_tag',
        names=['a', 'b'],
        range_=(2, 10),
        dtype=np.float32
    )

    chunk_spec2 = EncodingChunkSpec(
        tag='my_tag2',
        names=['a', 'b'],
        range_=(-2, 10),
        dtype=np.float32
    )

    chunk_extractor = ChunkExtractor(df, [chunk_spec, chunk_spec2])
    data = chunk_extractor.extract(2)

    logger.info(data)

    # Tag duplication.
    with pytest.raises(ValueError) as e:
        chunk_extractor = ChunkExtractor(df, [chunk_spec] * 2)

    logger.info(str(e.value))