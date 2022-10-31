import math
import numpy as np
import torch
import torch.nn as nn

from ..core import (
    ForecastingModule,
    Head,
)

from ..chunk import (
    EncodingChunkSpec,
    DecodingChunkSpec,
    LabelChunkSpec,
)

from ..layer import LeftPadding1D, Permute


class DilatedCNN(ForecastingModule):
    def __init__(
        self,
        hidden_size,
        encoding_length,
        decoding_length,
        target_names,
        nontarget_names,
        dilation_base,
        kernel_size,
        activation=nn.ELU,
        dropout_rate=0.0,
        lr=1e-3,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_options=None,
        loss_fn=None,
        metrics=None,
        head=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert kernel_size >= dilation_base

        self.encoding_length = encoding_length
        self.decoding_length = decoding_length

        if optimizer_options is None:
            self.hparams.optimizer_options = {}

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        n_outputs = len(target_names)
        n_features = len(nontarget_names) + n_outputs

        self.use_nontargets = n_outputs != n_features

        # Calculate the number of layers.
        # Formula is obtained from Darts.
        v = (encoding_length-1) * (dilation_base-1) / (kernel_size-1) + 1
        n_layers = math.ceil(math.log(v) / math.log(dilation_base))

        layers = []
        for i in range(n_layers):
            dilation = dilation_base**i
            padding_size = dilation * (kernel_size - 1)

            # (B, L', F)
            layers.append(LeftPadding1D(padding_size))
            # (B, F, L')
            layers.append(Permute(0, 2, 1))

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

            layers.append(activation())

            if dropout_rate > 1e-6:
                layers.append(nn.Dropout(p=dropout_rate))

            layers.append(Permute(0, 2, 1))

        # (B, H, L).
        self.body = nn.Sequential(*layers)

        if head is not None:
            self.head = head
        else:
            self.head = Head(
                tag='targets',
                output_module=nn.Linear(hidden_size, n_outputs),
                loss_fn=loss_fn,
                metrics=metrics,
            )

    def encode(self, inputs):
        # (B, L, F).
        if self.use_nontargets:
            x = torch.cat([
                inputs['encoding.targets'],
                inputs['encoding.nontargets']
            ], dim=2)
        else:
            x = inputs['encoding.targets']

        return {'x': x}

    def decode(self, inputs):
        # F = T + C.
        # (B, L, F)
        x = inputs['x']

        # (B, L, C).
        if self.use_nontargets:
            c = inputs['decoding.nontargets']

        self.head.reset()
        for i in range(self.decoding_length):
            # (B, 1, H).
            h = self.body(x)[:, -1:, :]
            # (B, 1, n_outputs).
            y = self.head(h)

            if i+1 == self.decoding_length:
                break
            # (B, 1, F).
            if self.use_nontargets:
                z = torch.cat([y, c[:, i:i+1, :]], dim=2)
            else:
                z = y

            # Concat on time dimension.
            # Remove first time step of previous input x
            # and append predicted value.
            x = torch.cat([
                x[:, 1:, :], z
            ], dim=1)

        outputs = self.head.get_outputs()

        return outputs

    def make_chunk_specs(self):
        E = self.encoding_length
        D = self.decoding_length

        chunk_specs = [
            EncodingChunkSpec(
                tag='targets',
                names=self.hparams.target_names,
                range_=(0, E),
                dtype=np.float32
            ),
            LabelChunkSpec(
                tag='targets',
                names=self.hparams.target_names,
                range_=(E, E+D),
                dtype=np.float32,
            ),
        ]

        if self.use_nontargets:
            chunk_specs += [
                EncodingChunkSpec(
                    tag='nontargets',
                    names=self.hparams.nontarget_names,
                    range_=(1, E+1),
                    dtype=np.float32,
                ),
                DecodingChunkSpec(
                    tag='nontargets',
                    names=self.hparams.nontarget_names,
                    range_=(E+1, E+D),
                    dtype=np.float32,
                ),
            ]

        return chunk_specs

    def configure_optimizers(self):
        return self.hparams.optimizer(
            self.parameters(),
            lr=self.hparams.lr,
            **self.hparams.optimizer_options,
        )