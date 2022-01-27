import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class MultiStepTransformerModel(nn.Module):
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
        ):
        super().__init__()
        self.n_encoder_features = n_encoder_features
        self.n_decoder_features = n_decoder_features
        self.encoding_length = encoding_length
        self.decoding_length = decoding_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.n_outputs = n_outputs

        self.encoder_d_matching_layer = nn.Linear(
            in_features=n_encoder_features,
            out_features=d_model,
        )

        self.decoder_d_matching_layer = nn.Linear(
            in_features=n_decoder_features,
            out_features=d_model,
        )

        self.past_pos_embedding = nn.Embedding(
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
        pos = self.past_pos_embedding(
            self.generate_range(self.encoding_length)
        )

        # B x L_past x d_model.
        x = x + pos

        # L_past x B x d_model.
        x = x.permute((1, 0, 2))

        # L_past x B x d_model.
        memory = self.encoder(x)

        return memory

    def decode(self, inputs, memory):
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

        return y

    def forward(self, inputs):
        memory = self.encode(inputs)
        y = self.decode(inputs, memory)

        return y

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

    @property
    def device(self):
        return next(self.parameters()).device


class MultiStepTransformerModelSystem(pl.LightningModule):
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
            lr
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiStepTransformerModel(
            n_encoder_features=n_encoder_features,
            n_decoder_features=n_decoder_features,
            encoding_length=encoding_length,
            decoding_length=decoding_length,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            n_outputs=n_outputs,
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _evaluate_loss(self, batch):
        y = self.model(batch)
        loss = self.loss_fn(y, batch['label.targets'])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._evaluate_loss(batch)
        self.log('loss/training', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._evaluate_loss(batch)
        self.log('loss/validation', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

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