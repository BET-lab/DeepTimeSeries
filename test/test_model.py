import sys
sys.path.append('..')

import pytest
import numpy as np
import pandas as pd
import pytorch_lightning as pl

import deep_time_series as dts
import deep_time_series.model
import logging
logger = logging.getLogger('test')

from torch.utils.data import DataLoader


@pytest.fixture()
def data():
    return pd.DataFrame(
        data={
            'a': np.arange(100),
            'b': np.arange(100)*2,
        }
    )


@pytest.fixture()
def trainer():
    return pl.Trainer(
        max_epochs=1,
    )


def make_loader(model, data):
    dataset = dts.TimeSeriesDataset(
        data, model.make_chunk_specs()
    )

    return DataLoader(dataset, batch_size=8)


def test_mlp(data, trainer):
    model = dts.model.MLP(
        hidden_size=4,
        encoding_length=4,
        decoding_length=3,
        target_names=['a'],
        nontarget_names=['b'],
        n_hidden_layers=2,
    )

    loader = make_loader(model, data)

    try:
        trainer.fit(
            model,
            train_dataloaders=loader,
            val_dataloaders=loader,
        )
        trainer.test(model, dataloaders=loader)
    except Exception as e:
        assert False, f'Model training fails: {e}'


# TODO: Add test on other models.