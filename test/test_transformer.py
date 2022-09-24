import sys
from typing import Type

from deep_time_series.transform import (
    _merge_data_frames,
    _split_data_frames,
    ColumnTransformer
)

sys.path.append('..')

import pytest

import logging
logger = logging.getLogger('test')

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


def test_transform():
    df = pd.DataFrame(data={
        'a': np.arange(10),
        'b': np.arange(10, 20),
        'c': np.arange(10)**2
    })

    dfs = _merge_data_frames(df)

    assert len(dfs) == 10
    logger.debug(dfs.columns)

    dfs = _merge_data_frames([df, df])

    assert len(dfs) == 20
    logger.debug(dfs.columns)

    dfs = _split_data_frames(dfs)

    assert len(dfs) == 2
    logger.debug(len(dfs))

    transform = ColumnTransformer(
        transformer_tuples=[
            (MinMaxScaler(), ['a', 'b'])
        ]
    )

    scaled_df = transform.fit_transform(df)
    assert np.allclose(np.arange(10)/9, scaled_df['a'])
    assert np.allclose(np.arange(10)/9, scaled_df['b'])

    with pytest.raises(KeyError):
        # No key 'c' exists.
        # Because no scaler for 'c' is defined.
        np.allclose(np.arange(10)**2/81, scaled_df['c'])

    scaled_dfs = transform.fit_transform([df, df])
    assert np.allclose(np.arange(10)/9, scaled_dfs[0]['a'])
    assert np.allclose(np.arange(10)/9, scaled_dfs[0]['b'])

    with pytest.raises(KeyError):
        # No key 'c' exists.
        # Because no scaler for 'c' is defined.
        np.allclose(np.arange(10)**2/81, scaled_dfs[0]['c'])
