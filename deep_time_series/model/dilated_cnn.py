import math
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


class LeftPadding1D(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x):
        # x: (B, C, L).
        B = x.size(0)
        C = x.size(1)

        padding = torch.zeros(B, C, self.padding_size).to(x.device)

        y = torch.cat([padding, x], axis=2)

        return y


class DilatedCNN(ForecastingModule):
    def __init__(
            self,
            n_features,
            hidden_size,
            encoding_length,
            dilation_base,
            kernel_size,
            activation,
            n_outputs,
            lr,
            loss_fn,
            head=None,
        ):
        super().__init__()
        self.save_hyperparameters()

        assert kernel_size >= dilation_base

        # Calculate the number of layers.
        # Formula is obtained from Darts.
        v = (encoding_length - 1) * (dilation_base - 1) / (kernel_size - 1) + 1
        n_layers = math.ceil(math.log(v) / math.log(dilation_base))

        layers = []
        for i in range(n_layers):
            dilation = dilation_base**i
            padding_size = dilation * (kernel_size - 1)

            layers.append(LeftPadding1D(padding_size))

            if i == 0:
                layers.append(nn.Conv1d(
                    n_features, hidden_size,
                    kernel_size=kernel_size, dilation=dilation,
                ))
            else:
                layers.append(nn.Conv1d(
                    hidden_size, hidden_size,
                    kernel_size=kernel_size, dilation=dilation,
                ))

            layers.append(activation)

        # (B, H, L).
        self.body = nn.Sequential(*layers)
        if head is None:
            self.head = nn.Linear(hidden_size, n_outputs)
        else:
            self.head = head

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
        L = c.size(1)

        ys = []
        for i in range(L):
            # (B, F, L).
            permuted_x = x.permute(0, 2, 1)
            # (B, H).
            y = self.body(permuted_x)[:, :, -1]
            # (B, 1, H).
            y = y.unsqueeze(1)
            # (B, 1, n_outputs).
            y = self.head(y)

            ys.append(y)

            # (B, 1, F).
            z = torch.cat([y, c[:, i:i+1, :]], dim=2)
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