import math

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .forecasting_module import ForecastingModule


class MultiStepTransformer(ForecastingModule):
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
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, n_outputs),
        )

    def encode(self, inputs):
        # L: encoding length.
        # all_input: (B, L, F).
        all_input = torch.cat([
            inputs['encoding.targets'],
            inputs['encoding.covariates']
        ], dim=2)

        x = self.encoder_d_matching_layer(all_input)

        x = self.positional_encoding(x)

        # (L, B, d_model).
        x = x.permute((1, 0, 2))

        # (L, B, d_model).
        memory = self.encoder(x)

        return {
            'memory': memory
        }

    def decode_train(self, inputs):
        # L: decoding_length
        memory = inputs['memory']

        all_input = inputs['decoding.covariates']
        x = self.decoder_d_matching_layer(all_input)

        # (B, L, d_model).
        L_future = all_input.shape[1]

        x = self.positional_encoding(x)

        # (L, B, d_model).
        x = x.permute((1, 0, 2))

        # (L, B, d_model).
        tgt_mask = self.generate_square_subsequent_mask(x.size(0))
        x = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)

        # (L, B, n_outputs).
        y = self.head(x)

        # (B, L, x n_outputs).
        y = y.permute((1, 0, 2))

        return {
            'label.targets': y
        }

    def decode_eval(self, inputs):
        return self.decode_train(inputs)

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

    def configure_callbacks(self):
        return [
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