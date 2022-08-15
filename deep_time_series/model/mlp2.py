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

class MLP2(ForecastingModule):
    def __init__(
        self,
        n_features,
        hidden_size,
        encoding_length,
        n_hidden_layers,
        activation,
        n_outputs,
        dropout_rate,
        lr,
        loss_fn,
        head=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        size = hidden_size * encoding_length
        layers = [
            nn.Linear(n_features*encoding_length, hidden_size), activation
        ]
        for i in range(n_hidden_layers):
            if dropout_rate > 1e-6:
                layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)

        if dropout_rate > 1e-6:
            layers.append(nn.Dropout(p=dropout_rate))

        if head is None:
            self.head = nn.Linear(hidden_size, n_outputs)
        else:
            self.head = head

        self.body = nn.Sequential(*layers)

    def encode(self, inputs):
       # (B, L, F).
        x = torch.cat([
            inputs['encoding.targets'],
            inputs['encoding.covariates']
        ], dim=2)

        return {'x': x}

    def decode_eval(self, inputs):
        # F = T + C.
        # (B, L, F)
        x = inputs['x']

        # (B, L, C).
        c = inputs['decoding.covariates']

        B = c.size(0)
        L = c.size(1)

        EL = self.hparams.encoding_length

        ys = []
        for i in range(L):
            # (B, L*F)
            x = x.view(B, -1)
            # (B, 1, n_outputs).
            y = self.body(x)
            y = y.unsqueeze(1)
            y = self.head(y)
            ys.append(y)

            # (B, 1, F).
            z = torch.cat([y, c[:, i:i+1, :]], dim=2)
            # (B, EL, F).
            x = x.view(B, EL, -1)
            # (B, L, F).
            x = torch.cat([
                x[:, 1:, :], z
            ], dim=1)

        y = torch.cat(ys, dim=1)

        return {
            'label.targets': y,
        }

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
                dtype=np.float32,
                shift=1,
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
                dtype=np.float32,
            ),
        ]

        return chunk_specs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)