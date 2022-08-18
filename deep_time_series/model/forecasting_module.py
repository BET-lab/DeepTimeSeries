import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..util import merge_dicts


class ForecastingModule(pl.LightningModule):
    def __init__(self):
        """Base class of all forecasting modules.
        """
        super().__init__()

        self.__encoding_length = None
        self.__decoding_length = None

    @property
    def encoding_length(self) -> int:
        """Encoding length."""
        if self.__encoding_length is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.encoding_length'
            )
        else:
            return self.__encoding_length

    @encoding_length.setter
    def encoding_length(self, value: int):
        if not isinstance(value, int):
            raise TypeError(
                f'Invalid type for "encoding_length": {type(value)}'
            )
        self.__encoding_length = value

    @property
    def decoding_length(self) -> int:
        if self.__decoding_length is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.decoding_length'
            )
        else:
            return self.__decoding_length

    @decoding_length.setter
    def decoding_length(self, value: int):
        if not isinstance(value, int):
            raise TypeError(
                f'Invalid type for "decoding_length": {type(value)}'
            )
        self.__decoding_length = value

    def encode(self, inputs):
        raise NotImplementedError(
            f'Define {self.__class__.__name__}.encode()'
        )

    def decode_train(self, inputs):
        return self.decode_eval(inputs)

    def decode_eval(self, inputs):
        NotImplementedError(
            f'Define {self.__class__.__name__}.decode()'
        )

    def evaluate_loss(self, batch):
        outputs = self(batch)
        loss = self.hparams.loss_fn(outputs, batch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.evaluate_loss(batch)
        self.log('loss/training', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.evaluate_loss(batch)
        self.log('loss/validation', loss)
        self.log('hp_metric', loss)

    def test_step(self, batch, batch_idx):
        loss = self.evaluate_loss(batch)
        self.log('loss/test', loss)

    def decode(self, inputs):
        if self.training:
            return self.decode_train(inputs)
        else:
            return self.decode_eval(inputs)

    def forward(self, inputs):
        encoder_outputs = self.encode(inputs)
        decoder_inputs = merge_dicts(
            [inputs, encoder_outputs]
        )
        outputs = self.decode(decoder_inputs)

        return outputs

    def make_chunk_specs(self, *args, **kwargs):
        pass