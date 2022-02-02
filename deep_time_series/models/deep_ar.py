import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ..utils import merge_dicts


class _DeepAR(nn.Module):
    def __init__(
            self,
            n_features,
            encoding_length,
            decoding_length,
            hidden_size,
            n_layers,
            n_outputs,
            rnn_class,
            dropout_rate,
        ):
        super().__init__()
        self.n_features = n_features
        self.encoding_length = encoding_length
        self.decoding_length = decoding_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.rnn_class = rnn_class

        self.encoder = rnn_class(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.decoder = self.encoder

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, n_outputs),
        )

    def encode(self, inputs):
        # all_input: B x L_past X C.
        all_input = torch.cat([
            inputs['encoding.targets'],
            inputs['encoding.covariates']
        ], dim=2)

        # Don't use last time step.
        x = all_input[:, :-1, :]

        # B x L_past x hidden_size.
        x, hidden_state = self.encoder(x)

        return {
            'y': x,
            'memory': hidden_state,
        }

    def decode(self, inputs):
        if self.training:
            return self.decode_train(inputs)
        else:
            return self.decode_eval(inputs)

    def decode_train(self, inputs):
        all_input = torch.cat([
            inputs['decoding.targets'],
            inputs['decoding.covariates']
        ], dim=2)

        x = all_input

        hidden_state = inputs['memory']

        # B x L_future x hidden_size.
        x, hidden_state = self.decoder(x, hidden_state)

        # L_future x B x n_outputs.
        y = self.head(x)

        return {
            'label.targets': y,
            'hidden_state': hidden_state,
        }

    def decode_eval(self, inputs):
        # decoding.covariates: (B, L, F).
        L = inputs['decoding.covariates'].shape[1]

        # decoding.targets: (B, 1, F).
        y = inputs['decoding.targets'][:, :1, :]
        c = inputs['decoding.covariates']

        hidden_state = inputs['memory']

        ys = []
        for i in range(L):
            x = torch.cat([y, c[:, i:i+1, :]], dim=2)
            y, hidden_state = self.decoder(x, hidden_state)
            y = self.head(y)
            ys.append(y)

        y = torch.cat(ys, dim=1)

        return {
            'label.targets': y,
            'hidden_state': hidden_state,
        }

    def forward(self, inputs):
        encoder_outputs = self.encode(inputs)
        decoder_inputs = merge_dicts(
            [inputs, encoder_outputs], ignore_keys='y'
        )
        outputs = self.decode(decoder_inputs)

        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


class DeepAR(pl.LightningModule):
    def __init__(
            self,
            n_features,
            encoding_length,
            decoding_length,
            hidden_size,
            n_layers,
            n_outputs,
            rnn_class,
            dropout_rate,
            lr,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = _DeepAR(
            n_features=n_features,
            encoding_length=encoding_length,
            decoding_length=decoding_length,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_outputs=n_outputs,
            rnn_class=rnn_class,
            dropout_rate=dropout_rate,
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _evaluate_loss(self, batch):
        outputs = self.model(batch)
        loss = self.loss_fn(
            outputs['label.targets'],
            batch['label.targets']
        )

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