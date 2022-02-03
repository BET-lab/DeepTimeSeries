import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .forecasting_module import ForecastingModule


class DeepAR(ForecastingModule):
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
            teacher_forcing,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn = nn.MSELoss()

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

    def decode_train(self, inputs):
        if not self.hparams.teacher_forcing:
            return self.decode_eval(inputs)

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

    def evaluate_loss(self, batch):
        outputs = self(batch)
        loss = self.loss_fn(
            outputs['label.targets'],
            batch['label.targets']
        )

        return loss

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