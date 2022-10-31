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

from ..layer import PositionalEncoding


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