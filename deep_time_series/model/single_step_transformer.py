import math

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ..core import ForecastingModule
from ..data import (
    EncodingChunkSpec,
    DecodingChunkSpec,
    LabelChunkSpec,
)


class SingleStepTransformer(ForecastingModule):
    def __init__(
        self,
        n_encoder_features,
        n_decoder_features,
        d_model,
        n_heads,
        n_layers,
        dim_feedforward,
        n_outputs,
        dropout_rate,
        lr,
        loss_fn,
        teacher_forcing_rate,
        head=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_d_matching_layer = nn.Linear(
            in_features=n_encoder_features,
            out_features=d_model,
        )

        self.decoder_d_matching_layer = nn.Linear(
            in_features=n_decoder_features,
            out_features=d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=5000,
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

        if head is None:
            self.head = nn.Linear(d_model, n_outputs)
        else:
            self.head = head

    def encode(self, inputs):
        # L: encoding length.
        # all_input: (B, L, F).
        all_input = torch.cat([
            inputs['encoding.targets'],
            inputs['encoding.covariates']
        ], dim=2)

        x = self.encoder_d_matching_layer(all_input)

        x = self.positional_encoding(x)

        # (B, L, d_model).
        memory = self.encoder(x)

        return {
            'memory': memory
        }

    def decode_eval(self, inputs):
        # decoding.covariates: (B, L, F).
        c = inputs['decoding.covariates']
        L = c.shape[1]

        # (B, 1, F). Last time step.
        y = inputs['encoding.targets'][:, -1:, :]
        memory = inputs['memory']

        ys = [y]
        for i in range(L):
            # (B, i+1, F).
            y = torch.cat(ys, dim=1)

            # (B, i+1, F).
            all_input = torch.cat([
                y, c[:, :i+1, :],
            ], dim=2)

            x = self.decoder_d_matching_layer(all_input)
            x = self.positional_encoding(x)

            x = self.decoder(tgt=x, memory=memory)

            # (B, i+1, n_outputs).
            y = self.head(x)[:, -1:, :]
            ys.append(y)

        y = torch.cat(ys[1:], dim=1)

        return {
            'label.targets': y,
        }

    def decode_train(self, inputs):
        if np.random.random() > self.hparams.teacher_forcing_rate:
            return self.decode_eval(inputs)

        # L: decoding_length
        memory = inputs['memory']

        all_input = torch.cat([
            inputs['decoding.targets'],
            inputs['decoding.covariates']
        ], dim=2)

        x = self.decoder_d_matching_layer(all_input)
        x = self.positional_encoding(x)

        # (B, L, d_model).
        tgt_mask = self.generate_square_subsequent_mask(x.size(1))
        x = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)

        # (B, L, n_outputs).
        y = self.head(x)

        return {
            'label.targets': y
        }

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence.
            The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(
            torch.full((sz, sz), float('-inf')), diagonal=1
        ).to(self.device)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
            ),

            DecodingChunkSpec(
                tag='targets',
                names=target_names,
                dtype=np.float32,
                shift=-1,
            ),
            DecodingChunkSpec(
                tag='covariates',
                names=covariate_names,
                dtype=np.float32
            ),

            LabelChunkSpec(
                tag='targets',
                names=target_names,
                dtype=np.float32
            ),
        ]

        return chunk_specs

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