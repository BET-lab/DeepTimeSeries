from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.distributions

import pytorch_lightning as pl

from ..util import merge_dicts


class BaseHead(nn.Module):
    SPECIAL_ATTRIBUTES = (
        'tag',
        'loss_weight',
    )

    def __init__(self):
        """Base class of all Head classes."""
        super().__init__()
        self.__tag = None
        self.__loss_weight = None

    def __setattr__(self, name, value):
        if name in BaseHead.SPECIAL_ATTRIBUTES:
            return object.__setattr__(self, name, value)
        else:
            return super().__setattr__(name, value)

    @property
    def tag(self) -> str:
        """Tag for a head. Prefix 'head.' is added automatically."""
        if self.__tag is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.tag'
            )
        else:
            return self.__tag

    @tag.setter
    def tag(self, value: str):
        if not isinstance(value, str):
            raise TypeError(
                f'Invalid type for "tag": {type(value)}'
            )

        if not value.startswith('head.'):
            value = f'head.{value}'

        self.__tag = value

    @property
    def loss_weight(self) -> float:
        """Loss weight for loss calculations."""
        if self.__loss_weight is None:
            self.__loss_weight = 1.0
        return self.__loss_weight

    @loss_weight.setter
    def loss_weight(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError(
                f'Invalid type for "loss_weight": {type(value)}'
            )
        elif value < 0:
            raise ValueError('loss_weight < 0')

        self.__loss_weight = value

    @property
    def label_tag(self) -> str:
        """Tag of target label. If the tag of head is "head.my_tag" then
        label_tag is "label.my_tag".
        """
        return f'label.{self.tag[5:]}'

    def forward(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError(
            f'Define {self.__class__.__name__}.forward()'
        )

    def get_outputs(self) -> dict[str, Any]:
        raise NotImplementedError(
            f'Define {self.__class__.__name__}.get_outputs()'
        )

    def reset_outputs(self):
        raise NotImplementedError(
            f'Define {self.__class__.__name__}.get_outputs()'
        )

    def calculate_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any]
    ) -> torch.Tensor:
        raise NotImplementedError(
            f'Define {self.__class__.__name__}.calculate_loss()'
        )


class Head(BaseHead):
    def __init__(
        self,
        tag: str,
        output_module: nn.Module,
        loss_fn: Callable,
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.tag = tag
        self.output_module = output_module
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight

        self._ys = []

    def forward(self, inputs: Any) -> torch.Tensor:
        y = self.output_module(inputs)
        self._ys.append(y)
        return y

    def get_outputs(self):
        return {
            self.tag: torch.cat(self._ys, dim=1)
        }

    def reset_outputs(self):
        self._ys = []

    def calculate_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any]
    ) -> torch.Tensor:
        return self.loss_fn(outputs[self.tag], batch[self.label_tag])


class DistributionHead(BaseHead):
    def __init__(self,
        tag,
        distribution: torch.distributions.Distribution,
        in_features,
        out_features,
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.tag = tag
        self.loss_weight = loss_weight

        linears = {}
        transforms = {}
        for k, v in distribution.arg_constraints.items():
            # 'logits' is prefered.
            if k == 'probs':
                continue

            linears[k] = nn.Linear(in_features, out_features)
            transforms[k] = torch.distributions.transform_to(v)

        self.distribution = distribution

        self.linears = nn.ModuleDict(linears)
        self.transforms = transforms

        self._outputs = defaultdict(list)

    def forward(self, x):
        kwargs = {
            k: self.transforms[k](layer(x))
            for k, layer in self.linears.items()
        }

        m = self.distribution(**kwargs)
        y = m.sample()

        for k, v in kwargs.items():
            self._outputs[f'{self.tag}.{k}'].append(v)

        self._outputs[self.tag].append(y)

        return y

    def get_outputs(self):
        outputs = {}
        for k, v in self._outputs.items():
            outputs[k] = torch.cat(v, dim=1)

        return outputs

    def reset_outputs(self):
        self._outputs = defaultdict(list)

    def calculate_loss(
            self,
            outputs,
            batch
        ) -> torch.Tensor:
        kwargs = {
            k: outputs[f'{self.tag}.{k}']
            for k in self.linears.keys()
        }

        m = self.distribution(**kwargs)

        return -torch.mean(m.log_prob(batch[self.label_tag]))


class ForecastingModule(pl.LightningModule):
    SPECIAL_ATTRIBUTES = (
        'encoding_length',
        'decoding_length',
        'head',
        'heads',
    )

    def __init__(self):
        """Base class of all forecasting modules.
        """
        super().__init__()

        self.__encoding_length = None
        self.__decoding_length = None

        self.__heads = None

    def __setattr__(self, name, value):
        if name in ForecastingModule.SPECIAL_ATTRIBUTES:
            return object.__setattr__(self, name, value)
        else:
            return super().__setattr__(name, value)

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
    def heads(self) -> list[BaseHead]:
        if self.__heads is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.heads'
            )
        else:
            return self.__heads

    @heads.setter
    def heads(self, heads: list[BaseHead]):
        if not isinstance(heads, list):
            raise TypeError(
                f'Invalid type for "heads". {type(heads)}'
            )
        elif not all(isinstance(head, BaseHead) for head in heads):
            raise TypeError(
                f'Invalid type for "heads". {[type(v) for v in heads]}'
            )

        self.__heads = nn.ModuleList(heads)

    @property
    def head(self) -> BaseHead:
        if self.__heads is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.heads'
            )
        elif len(self.__heads) != 1:
            raise Exception('Multi-head model cannot use head.')
        else:
            return self.__heads[0]

    @head.setter
    def head(self, head: BaseHead):
        if not isinstance(head, BaseHead):
            raise TypeError(
                f'Invalid type for "heads". {type(head)}'
            )

        self.heads = [head]

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
            loss += head.loss_weight * head.calculate_loss(outputs, batch)

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