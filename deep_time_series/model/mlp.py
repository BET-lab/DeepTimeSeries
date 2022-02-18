import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .forecasting_module import ForecastingModule


class MLP(ForecastingModule):
    def __init__(
            self,
            n_features,
            hidden_size,
            encoding_length,
            n_hidden_layers,
            activation,
            n_outputs,
            lr,
            loss_fn,
        ):
        super().__init__()
        self.save_hyperparameters()

        size = hidden_size * encoding_length
        layers = [
            nn.Linear(n_features*encoding_length, size), activation
        ]
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(size, size))
            layers.append(activation)
        layers.append(nn.Linear(size, n_outputs))

        self.mlp = nn.Sequential(*layers)

    def encode(self, inputs):
       # (B, L, F).
        x = torch.cat([
            inputs['encoding.targets'],
            inputs['encoding.covariates']
        ], dim=2)

        return {'x': x}

    def decode_train(self, inputs):
        return self.decode_eval(inputs)

    def decode_eval(self, inputs):
        # F = T + C.
        # (B, L, F)
        x = inputs['x']

        # (B, L, C).
        c = inputs['decoding.covariates']

        B = c.size(0)
        L = c.size(1)

        ys = []
        for i in range(L):
            # (B, L*F)
            x = x.view(B, -1)
            # (B, 1, n_outputs).
            y = self.mlp(x).unsqueeze(1)
            ys.append(y)

            # (B, 1, F).
            z = torch.cat([y, c[:, i:i+1, :]], dim=2)
            # (B, 1, F).
            x = x.view(B, L, -1)
            x = torch.cat([
                x[:, 1:, :], z
            ], dim=1)

        y = torch.cat(ys, dim=1)

        return {
            'label.targets': y,
        }

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