import sys
from typing import Type

from deep_time_series.chunk import (
    ChunkInverter,
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

import torch

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


def test_chunk_inverter():
    chunk_spec = EncodingChunkSpec(
        tag='my_tag',
        names=['a', 'b'],
        range_=(2, 10),
        dtype=np.float32
    )

    chunk_specs = [chunk_spec]

    ci = ChunkInverter(chunk_specs=chunk_specs)


    assert isinstance(
        ci._convert_to_numpy(np.array([1, 2])), np.ndarray
    )

    value = torch.FloatTensor([1.0, 2.0])
    logger.debug(type(value))

    assert isinstance(
        ci._convert_to_numpy(value), np.ndarray
    )

    # Only 2 features.
    data = {
        'my_tag': torch.rand(size=(3, 7, 2)),
        'head.my_tag': torch.rand(size=(3, 7, 2)),
        'label.my_tag': torch.rand(size=(3, 7, 2)),
    }

    # names of 'encoding.my_tag' used as the core tag is 'my_tag'.
    df = ci.invert('my_tag', data['my_tag'])
    # names of 'encoding.my_tag' used as the core tag is 'my_tag'.
    df = ci.invert('head.my_tag', data['my_tag'])
    # names of 'encoding.my_tag' used as the core tag is 'my_tag'.
    df = ci.invert('label.my_tag', data['my_tag'])

    # names of 'encoding.my_tag' used as the tag is 'encoding.my_tag'.
    # Exact matching is occured here only.
    df = ci.invert('label.my_tag', data['my_tag'])

    # Suffixes also ignored.
    df = ci.invert('label.my_tag.loc', data['my_tag'])

    dfs = ci.invert_dict(data)