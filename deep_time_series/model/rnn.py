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


class RNN(ForecastingModule):
    def __init__(
        self,
        hidden_size,
        encoding_length,
        decoding_length,
        target_names,
        nontarget_names,
        n_layers,
        rnn_class,
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

        self.encoding_length = encoding_length
        self.decoding_length = decoding_length

        if optimizer_options is None:
            self.hparams.optimizer_options = {}

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        n_outputs = len(target_names)
        n_features = len(nontarget_names) + n_outputs

        self.use_nontargets = n_outputs != n_features

        self.encoder = rnn_class(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        # Use same model instance used in encoding process.
        self.decoder = self.encoder

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

        # (B, L, H).
        # For LSTM, hidden_state is a tuple: (h_0, c_0).
        h, memory = self.encoder(x)

        return {
            'h': h[:, -1:, :],
            'memory': memory,
        }

    def decode(self, inputs):
        self.head.reset()
        y = self.head(inputs['h'])

        if self.use_nontargets:
            c = inputs['decoding.nontargets']
            x = torch.cat([y, c[:, 0:1, :]], dim=2)
        else:
            x = y

        memory = inputs['memory']

        for i in range(1, self.decoding_length):
            h, memory = self.decoder(x, memory)
            y = self.head(h)

            if i+1 == self.decoding_length:
                break

            if self.use_nontargets:
                x = torch.cat([y, c[:, i:i+1, :]], dim=2)
            else:
                x = y

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