from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import pytorch_lightning as pl


class ForecastingModule(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, inputs):
        pass

    @abstractmethod
    def decode_train(self, inputs):
        pass

    @abstractmethod
    def decode_eval(self, inputs):
        pass

    @abstractmethod
    def evaluate_loss(self, batch):
        # return loss
        pass

    def training_step(self, batch, batch_idx):
        loss = self.evaluate_loss(batch)
        self.log('loss/training', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.evaluate_loss(batch)
        self.log('loss/validation', loss)

    def decode(self, inputs):
        if self.training:
            return self.decode_train(inputs)
        else:
            return self.decode_eval(inputs)

    @property
    def device(self):
        return next(self.parameters()).device