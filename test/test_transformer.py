import sys

from deep_time_series.transform import ColumnTransformer, _merge_data_frames

sys.path.append('..')

import logging

import pytest

logger = logging.getLogger('test')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def test_transform():
    df = pd.DataFrame(
        data={
            'a': np.arange(10),
            'b': np.arange(10, 20),
            'c': np.arange(10) ** 2,
        }
    )

    dfs = _merge_data_frames(df)

    assert len(dfs) == 10
    logger.debug(dfs.columns)

    dfs = _merge_data_frames([df, df])

    assert len(dfs) == 20
    logger.debug(dfs.columns)

    transform = ColumnTransformer(
        transformer_tuples=[(MinMaxScaler(), ['a', 'b'])]
    )

    scaled_df = transform.fit_transform(df)
    assert np.allclose(np.arange(10) / 9, scaled_df['a'])
    assert np.allclose(np.arange(10) / 9, scaled_df['b'])

    with pytest.raises(KeyError):
        # No key 'c' exists.
        # Because no scaler for 'c' is defined.
        np.allclose(np.arange(10) ** 2 / 81, scaled_df['c'])

    scaled_dfs = transform.fit_transform([df, df])
    assert np.allclose(np.arange(10) / 9, scaled_dfs[0]['a'])
    assert np.allclose(np.arange(10) / 9, scaled_dfs[0]['b'])

    with pytest.raises(KeyError):
        # No key 'c' exists.
        # Because no scaler for 'c' is defined.
        np.allclose(np.arange(10) ** 2 / 81, scaled_dfs[0]['c'])
