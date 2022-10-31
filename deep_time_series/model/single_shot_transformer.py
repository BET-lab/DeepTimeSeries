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


class SingleShotTransformer(ForecastingModule):
    def __init__(
        self,
        encoding_length,
        decoding_length,
        target_names,
        nontarget_names,
        d_model,
        n_heads,
        n_layers,
        dim_feedforward=None,
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

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        n_targets = len(target_names)
        n_nontargets = len(nontarget_names)
        n_features = n_nontargets + n_targets

        self.use_nontargets = n_nontargets > 0

        self.encoder_d_matching_layer = nn.Linear(
            in_features=n_features,
            out_features=d_model,
        )

        if self.use_nontargets:
            self.decoder_d_matching_layer = nn.Linear(
                in_features=n_nontargets,
                out_features=d_model,
            )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=max(encoding_length, decoding_length),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        if head is not None:
            self.head = head
        else:
            self.head = Head(
                tag='targets',
                output_module=nn.Linear(d_model, n_targets),
                loss_fn=loss_fn,
                metrics=metrics,
            )

    def encode(self, inputs):
        # L: encoding length.
        # all_input: (B, L, F).
        if self.use_nontargets:
            x = torch.cat([
                inputs['encoding.targets'],
                inputs['encoding.nontargets']
            ], dim=2)
        else:
            x = inputs['encoding.targets']


        x = self.encoder_d_matching_layer(x)
        x = self.positional_encoding(x)

        # (B, L, d_model).
        memory = self.encoder(x)

        return {
            'memory': memory
        }

    def decode(self, inputs):
        # L: decoding_length
        memory = inputs['memory']

        if self.use_nontargets:
            x = inputs['decoding.nontargets']
            x = self.decoder_d_matching_layer(x)
        else:
            # Same device will be used automatically.
            x = torch.zeros_like(memory)

        x = self.positional_encoding(x)

        # (B, L, d_model).
        tgt_mask = self.generate_square_subsequent_mask(x.size(1))
        x = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)

        # (B, L, n_outputs).
        self.head.reset()
        self.head(x)
        outputs = self.head.get_outputs()

        return outputs

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence.
            The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(
            torch.full((sz, sz), float('-inf')), diagonal=1
        ).to(self.device)

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
                    range_=(0, E),
                    dtype=np.float32,
                ),
                DecodingChunkSpec(
                    tag='nontargets',
                    names=self.hparams.nontarget_names,
                    range_=(E, E+D),
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

# Modified from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
# Modifications
# 1. Dropout functionality is removed.
# 2. Changed to batch_first format.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) \
            * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Change to batch_first format.
        pe = pe.permute((1, 0, 2))

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, F).
        x = x + self.pe[:, :x.size(1), :]
        return x