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

from ..layer import (
    Unsqueesze,
)


class MLP(ForecastingModule):
    def __init__(
        self,
        hidden_size,
        encoding_length,
        decoding_length,
        target_names,
        non_target_names,
        n_hidden_layers,
        activation=nn.ReLU,
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

        if optimizer_options is None:
            self.hparams.optimizer_options = {}

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        n_outputs = len(target_names)
        n_features = len(non_target_names) + n_outputs

        self.use_non_targets = n_outputs != n_features

        self.encoding_length = encoding_length
        self.decoding_length = decoding_length

        layers = [
            nn.Flatten(),
            nn.Linear(n_features*encoding_length, hidden_size),
            activation()
        ]
        for i in range(n_hidden_layers):
            if dropout_rate > 1e-6:
                layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())

        if dropout_rate > 1e-6:
            layers.append(nn.Dropout(p=dropout_rate))

        layers.append(Unsqueesze(1))

        if head is not None:
            self.head = head
        else:
            self.head = Head(
                tag='targets',
                output_module=nn.Linear(hidden_size, n_outputs),
                loss_fn=loss_fn,
                metrics=metrics,
            )

        self.body = nn.Sequential(*layers)

    def encode(self, inputs):
        # (B, L, F).
        if self.use_non_targets:
            x = torch.cat([
                inputs['encoding.targets'],
                inputs['encoding.non_targets']
            ], dim=2)
        else:
            x = inputs['encoding.targets']

        return {'x': x}

    def decode(self, inputs):
        # F = T + C.
        # (B, L, F)
        x = inputs['x']

        # (B, L, C).
        if self.use_non_targets:
            c = inputs['decoding.non_targets']

        B = x.size(0)
        L = x.size(1)

        EL = self.encoding_length

        for i in range(self.decoding_length):
            # (B, L*F)
            # x = x.view(B, -1)
            # (B, 1, n_outputs).
            y = self.body(x)
            # y = y.unsqueeze(1)
            y = self.head(y)

            if i+1 == self.decoding_length:
                break
            # (B, 1, F).
            if self.use_non_targets:
                z = torch.cat([y, c[:, i:i+1, :]], dim=2)
            else:
                z = y
            # z = y
            # (B, EL, F).
            x = x.view(B, EL, -1)
            # (B, L, F).
            x = torch.cat([
                x[:, 1:, :], z
            ], dim=1)

        outputs = self.head.get_outputs()
        self.head.reset()

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

        if self.use_non_targets:
            chunk_specs += [
                EncodingChunkSpec(
                    tag='non_targets',
                    names=self.hparams.non_target_names,
                    range_=(1, E+1),
                    dtype=np.float32,
                ),
                DecodingChunkSpec(
                    tag='non_targets',
                    names=self.hparams.non_target_names,
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