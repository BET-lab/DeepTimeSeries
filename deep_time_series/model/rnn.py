import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .forecasting_module import ForecastingModule
from ..data import (
    EncodingChunkSpec,
    DecodingChunkSpec,
    LabelChunkSpec,
)


class RNN(ForecastingModule):
    def __init__(
            self,
            n_features,
            hidden_size,
            n_layers,
            n_outputs,
            rnn_class,
            dropout_rate,
            lr,
            loss_fn,
            teacher_forcing,
            head=None,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = rnn_class(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.decoder = self.encoder

        if head is None:
            self.head = nn.Linear(hidden_size, n_outputs)
        else:
            self.head = head

    def encode(self, inputs):
        # L: encoding length.
        # all_input: (B, L, F).
        all_input = torch.cat([
            inputs['encoding.targets'],
            inputs['encoding.covariates']
        ], dim=2)

        # Don't use last time step.
        # (B, L-1, F).
        x = all_input[:, :-1, :]
        # (B, 1, F).
        last_x = all_input[:, -1:, :]

        # (B, L, H).
        y, hidden_state = self.encoder(x)

        return {
            'y': y,
            'memory': hidden_state,
            'last_x': last_x,
        }

    def decode_train(self, inputs):
        if not self.hparams.teacher_forcing:
            return self.decode_eval(inputs)

        all_input = torch.cat([
            inputs['decoding.targets'],
            inputs['decoding.covariates']
        ], dim=2)

        # Don't use last time.
        # (B, L-1, F).
        x = all_input[:, :-1, :]

        # Concat last of encoding input.
        # (B, 1, F).
        last_x = inputs['last_x']
        # (B, L, F).
        x = torch.cat([last_x, x], dim=1)

        hidden_state = inputs['memory']

        # (B, L, H).
        x, hidden_state = self.decoder(x, hidden_state)

        # (L, B, n_outputs).
        y = self.head(x)

        return {
            'label.targets': y,
            'hidden_state': hidden_state,
        }

    def decode_eval(self, inputs):
        # decoding.covariates: (B, L, F).
        c = inputs['decoding.covariates']
        L = c.shape[1]

        x = inputs['last_x']
        hidden_state = inputs['memory']

        ys = []
        for i in range(L):
            y, hidden_state = self.decoder(x, hidden_state)
            y = self.head(y)
            ys.append(y)
            x = torch.cat([y, c[:, i:i+1, :]], dim=2)

        y = torch.cat(ys, dim=1)

        return {
            'label.targets': y,
            'hidden_state': hidden_state,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def configure_callbacks(self):
        return  [
            EarlyStopping(
                monitor='loss/validation',
                mode='min',
                patience=50,
            ),
            ModelCheckpoint(
                monitor='loss/validation',
                mode='min',
            ),
        ]

    def make_chunk_specs(self, target_names, covariate_names):
        chunk_specs = [
            EncodingChunkSpec(
                tag='targets',
                names=target_names,
                dtype=np.float32
            ),
            EncodingChunkSpec(
                tag='covariates',
                names=covariate_names,
                dtype=np.float32, shift=1,
            ),
            DecodingChunkSpec(
                tag='targets',
                names=target_names,
                dtype=np.float32,
            ),
            DecodingChunkSpec(
                tag='covariates',
                names=covariate_names,
                dtype=np.float32,
                shift=1
            ),
            LabelChunkSpec(
                tag='targets',
                names=target_names,
                dtype=np.float32
            ),
        ]

        return chunk_specs