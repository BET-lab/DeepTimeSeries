from typing import Any, Callable

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..util import merge_dicts


class Head(nn.Module):
    def __init__(
        self,
        tag: str,
        output_module: nn.Module,
        loss_fn: Callable,
        weight: float = 1.0,
    ):
        super().__init__()

        self.tag = tag
        self.output_module = output_module
        self.loss_fn = loss_fn
        self.weight = weight

    def forward(self, inputs: Any) -> torch.Tensor:
        return self.output_module(inputs)

    def calculate_loss(
            self,
            outputs: dict[str, Any],
            batch: dict[str, Any]
        ) -> torch.Tensor:
        return self.loss_fn(outputs[self.tag], batch[self.tag])


class ForecastingModule(pl.LightningModule):
    def __init__(self):
        """Base class of all forecasting modules.
        """
        super().__init__()

        self.__encoding_length = None
        self.__decoding_length = None

        self.__heads = None

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
        elif value <= 0:
            raise ValueError(
                'Encoding length <= 0.'
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
        elif value <= 0:
            raise ValueError(
                'Decoding length <= 0.'
            )
        self.__decoding_length = value

    @property
    def heads(self) -> list[Head]:
        if self.__heads is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.heads'
            )
        else:
            return self.__heads

    @heads.setter
    def heads(self, heads: list[Head]):
        if not all(isinstance(head, Head) for head in heads):
            raise TypeError(
                f'Invalid type for "heads".'
            )

        self.__heads = nn.ModuleList(heads)

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            f'Define {self.__class__.__name__}.encode()'
        )

    def decode_train(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self.decode_eval(inputs)

    def decode_eval(self, inputs: dict[str, Any]) -> dict[str, Any]:
        NotImplementedError(
            f'Define {self.__class__.__name__}.decode()'
        )

    def calculate_loss(self, batch: dict[str, Any]) -> dict[str, Any]:
        outputs = self(batch)

        loss = 0
        for head in self.heads:
            loss += head.weight * head.calculate_loss(outputs, batch)

        return loss

    def training_step(
        self,
        batch: dict[str, Any], batch_idx: int
    ) -> dict[str, Any]:
        loss = self.calculate_loss(batch)
        self.log('loss/training', loss)

        return loss

    def validation_step(
        self,
        batch: dict[str, Any], batch_idx: int
    ) -> dict[str, Any]:
        loss = self.calculate_loss(batch)
        self.log('loss/validation', loss)
        self.log('hp_metric', loss)

    def test_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log('loss/test', loss)

    def decode(self, inputs):
        if self.training:
            return self.decode_train(inputs)
        else:
            return self.decode_eval(inputs)

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        encoder_outputs = self.encode(inputs)
        decoder_inputs = merge_dicts(
            [inputs, encoder_outputs]
        )
        outputs = self.decode(decoder_inputs)

        return outputs

    def make_chunk_specs(self):
        pass