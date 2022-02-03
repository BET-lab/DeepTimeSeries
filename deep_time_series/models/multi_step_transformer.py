import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ..utils import merge_dicts
from .forecasting_module import ForecastingModule


class MultiStepTransformer(ForecastingModule):
    def __init__(
            self,
            n_encoder_features,
            n_decoder_features,
            encoding_length,
            decoding_length,
            d_model,
            n_heads,
            n_layers,
            dim_feedforward,
            n_outputs,
            lr,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn = nn.MSELoss()

        self.encoder_d_matching_layer = nn.Linear(
            in_features=n_encoder_features,
            out_features=d_model,
        )

        self.decoder_d_matching_layer = nn.Linear(
            in_features=n_decoder_features,
            out_features=d_model,
        )

        self.encoding_pos_embedding = nn.Embedding(
            num_embeddings=encoding_length,
            embedding_dim=d_model,
        )

        self.future_pos_embedding = nn.Embedding(
            num_embeddings=decoding_length,
            embedding_dim=d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=dim_feedforward, dropout=0,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=dim_feedforward, dropout=0,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, n_outputs),
        )

    def encode(self, inputs):
        # all_input: B x L_past X C.
        all_input = torch.cat([
            inputs['encoding.targets'],
            inputs['encoding.covariates']
        ], dim=2)

        x = self.encoder_d_matching_layer(all_input)

        # B x L_past x d_model.
        pos = self.encoding_pos_embedding(
            self.generate_range(self.hparams.encoding_length)
        )

        # B x L_past x d_model.
        x = x + pos

        # L_past x B x d_model.
        x = x.permute((1, 0, 2))

        # L_past x B x d_model.
        memory = self.encoder(x)

        return {
            'memory': memory
        }

    def decode_train(self, inputs):
        memory = inputs['memory']

        all_input = inputs['decoding.covariates']
        x = self.decoder_d_matching_layer(all_input)

        # B x L_future x d_model.
        L_future = all_input.shape[1]

        pos = self.future_pos_embedding(
            self.generate_range(L_future)
        )

        # B x L_future x d_model.
        x = x + pos

        # L_future x B x d_model.
        x = x.permute((1, 0, 2))

        mask = self.generate_square_subsequent_mask(L_future)

        # L_future x B x d_model.
        x = self.decoder(tgt=x, memory=memory, tgt_mask=mask)

        # L_future x B x n_outputs.
        y = self.head(x)

        # B x L_future x n_outputs.
        y = y.permute((1, 0, 2))

        return {
            'label.targets': y
        }

    def decode_eval(self, inputs):
        return self.decode_train(inputs)

    def forward(self, inputs):
        encoder_outputs = self.encode(inputs)
        decoder_inputs = merge_dicts([inputs, encoder_outputs])
        outputs = self.decode(decoder_inputs)

        return outputs

    def evaluate_loss(self, batch):
        outputs = self(batch)
        loss = self.loss_fn(
            outputs['label.targets'],
            batch['label.targets']
        )

        return loss

    def generate_range(self, length):
        range_ = torch.arange(0, length)
        return range_.to(device=self.device, dtype=torch.long)

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