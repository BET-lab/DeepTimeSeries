from collections import defaultdict
from typing import Any, Callable

import pytorch_lightning as pl
import torch
import torch.distributions
import torch.nn as nn
from torchmetrics import Metric, MetricCollection

from .util import merge_dicts


class MetricModule(nn.Module):
    def __init__(
        self, tag: str, metrics: Metric | list[Metric] | dict[str, Metric]
    ):
        """Module for metrics. It's a wrapper of MetricCollections for training,
        validation and test stage. It's used in BaseHead class. In general,
        it's not used directly by users. """

        super().__init__()

        self.tag = tag

        if tag.startswith('head.'):
            self.head_tag = tag
            self.label_tag = f'label.{tag[5:]}'
        else:
            self.head_tag = f'head.{tag}'
            self.label_tag = f'label.{tag}'

        metrics = MetricCollection(metrics)

        self._metric_dict = nn.ModuleDict(
            {
                '_train': metrics.clone(prefix=f'train/{tag}.'),
                '_val': metrics.clone(prefix=f'val/{tag}.'),
                '_test': metrics.clone(prefix=f'test/{tag}.'),
            }
        )

    def forward(self, outputs, batch, stage):
        return self._metric_dict[f'_{stage}'](
            outputs[self.head_tag], batch[self.label_tag]
        )

    def compute(self, stage):
        return self._metric_dict[f'_{stage}'].compute()

    def update(self, outputs, batch, stage):
        self._metric_dict[f'_{stage}'].update(
            outputs[self.head_tag], batch[self.label_tag]
        )

    def reset(self, stage):
        self._metric_dict[f'_{stage}'].reset()


class BaseHead(nn.Module):
    _SPECIAL_ATTRIBUTES = ('tag', 'loss_weight', 'metrics')

    def __init__(self):
        """Base class of all Head classes. Head is a module that takes the
        output of the last layer of body and produces the output of the model.
        It also calculates the loss and metrics. User have to define following
        attributes:

        * tag: str
            Tag for a head. Prefix 'head.' is added automatically.
        * loss_weight: float
            Loss weight for loss calculations.
        * metrics: Metric | list[Metric] | dict[str, Metric]
            Metrics for training, validation and test stage.

        User have to define following methods:

        * forward(self, inputs: Any) -> torch.Tensor
            Forward method of the head.
        * get_outputs(self) -> dict[str, Any]
            Get outputs of the head. It produces a dictionary of outputs from
            internal state of the head.
        * reset(self)
            Reset the internal states of the head.
        * calculate_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor
            Calculate loss of the head.
        """
        super().__init__()
        self._tag = None
        self._loss_weight = None
        self._metrics = None

    def __setattr__(self, name, value):
        if name in BaseHead._SPECIAL_ATTRIBUTES:
            return object.__setattr__(self, name, value)
        else:
            return super().__setattr__(name, value)

    @property
    def tag(self) -> str:
        """Tag for a head. Prefix 'head.' is added automatically."""
        if self._tag is None:
            raise NotImplementedError(f'Define {self.__class__.__name__}.tag')
        else:
            return self._tag

    @tag.setter
    def tag(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f'Invalid type for "tag": {type(value)}')

        if not value.startswith('head.'):
            value = f'head.{value}'

        self._tag = value

    @property
    def metrics(self) -> MetricModule:
        if self._metrics is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.metrics'
            )
        else:
            return self._metrics

    @metrics.setter
    def metrics(self, value: Metric | list[Metric] | dict[str, Metric]):
        metric_module = MetricModule(tag=self.tag, metrics=value)
        self._metrics = metric_module

    @property
    def has_metrics(self):
        return self._metrics is not None

    @property
    def loss_weight(self) -> float:
        """Loss weight for loss calculations."""
        if self._loss_weight is None:
            self._loss_weight = 1.0
        return self._loss_weight

    @loss_weight.setter
    def loss_weight(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError(f'Invalid type for "loss_weight": {type(value)}')
        elif value < 0:
            raise ValueError('loss_weight < 0')

        self._loss_weight = value

    @property
    def label_tag(self) -> str:
        """Tag of target label. If the tag of head is "head.my_tag" then
        label_tag is "label.my_tag".
        """
        return f'label.{self.tag[5:]}'

    def forward(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError(f'Define {self.__class__.__name__}.forward()')

    def get_outputs(self) -> dict[str, Any]:
        raise NotImplementedError(
            f'Define {self.__class__.__name__}.get_outputs()'
        )

    def reset(self):
        raise NotImplementedError(f'Define {self.__class__.__name__}.reset()')

    def calculate_loss(
        self, outputs: dict[str, Any], batch: dict[str, Any]
    ) -> torch.Tensor:
        raise NotImplementedError(
            f'Define {self.__class__.__name__}.calculate_loss()'
        )


class Head(BaseHead):
    def __init__(
        self,
        tag: str,
        output_module: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        loss_weight: float = 1.0,
        metrics: Metric | list[Metric] | dict[str, Metric] = None,
    ):
        super().__init__()

        self.tag = tag
        self.output_module = output_module
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight

        if metrics is not None:
            self.metrics = metrics

        self._ys = []

    def forward(self, inputs: Any) -> torch.Tensor:
        y = self.output_module(inputs)
        self._ys.append(y)
        return y

    def get_outputs(self):
        return {self.tag: torch.cat(self._ys, dim=1)}

    def reset(self):
        self._ys = []

    def calculate_loss(
        self, outputs: dict[str, Any], batch: dict[str, Any]
    ) -> torch.Tensor:
        return self.loss_fn(outputs[self.tag], batch[self.label_tag])


class DistributionHead(BaseHead):
    def __init__(
        self,
        tag: str,
        distribution: torch.distributions.Distribution,
        in_features: int,
        out_features: int,
        loss_weight: float = 1.0,
        metrics: Metric | list[Metric] | dict[str, Metric] = None,
    ):
        super().__init__()

        self.tag = tag
        self.loss_weight = loss_weight
        if metrics is not None:
            self.metrics = metrics

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
            k: self.transforms[k](layer(x)) for k, layer in self.linears.items()
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

    def reset(self):
        self._outputs = defaultdict(list)

    def calculate_loss(self, outputs, batch) -> torch.Tensor:
        kwargs = {k: outputs[f'{self.tag}.{k}'] for k in self.linears.keys()}

        m = self.distribution(**kwargs)

        return -torch.mean(m.log_prob(batch[self.label_tag]))


class ForecastingModule(pl.LightningModule):
    _SPECIAL_ATTRIBUTES = (
        'encoding_length',
        'decoding_length',
        'head',
        'heads',
    )

    def __init__(self):
        """Base class of all forecasting modules."""
        super().__init__()

        self._encoding_length = None
        self._decoding_length = None

        self._heads = None

    def __setattr__(self, name, value):
        if name in ForecastingModule._SPECIAL_ATTRIBUTES:
            return object.__setattr__(self, name, value)
        else:
            return super().__setattr__(name, value)

    @property
    def encoding_length(self) -> int:
        """Encoding length."""
        if self._encoding_length is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.encoding_length'
            )
        else:
            return self._encoding_length

    @encoding_length.setter
    def encoding_length(self, value: int):
        if not isinstance(value, int):
            raise TypeError(
                f'Invalid type for "encoding_length": {type(value)}'
            )
        elif value <= 0:
            raise ValueError('Encoding length <= 0.')
        self._encoding_length = value

    @property
    def decoding_length(self) -> int:
        if self._decoding_length is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.decoding_length'
            )
        else:
            return self._decoding_length

    @decoding_length.setter
    def decoding_length(self, value: int):
        if not isinstance(value, int):
            raise TypeError(
                f'Invalid type for "decoding_length": {type(value)}'
            )
        elif value <= 0:
            raise ValueError('Decoding length <= 0.')
        self._decoding_length = value

    @property
    def heads(self) -> list[BaseHead]:
        if self._heads is None:
            raise NotImplementedError(f'Define {self.__class__.__name__}.heads')
        else:
            return self._heads

    @heads.setter
    def heads(self, heads: list[BaseHead]):
        if not isinstance(heads, list):
            raise TypeError(f'Invalid type for "heads". {type(heads)}')
        elif not all(isinstance(head, BaseHead) for head in heads):
            raise TypeError(
                f'Invalid type for "heads". {[type(v) for v in heads]}'
            )

        self._heads = nn.ModuleList(heads)

    @property
    def head(self) -> BaseHead:
        if self._heads is None:
            raise NotImplementedError(f'Define {self.__class__.__name__}.heads')
        elif len(self._heads) != 1:
            raise Exception('Multi-head model cannot use head.')
        else:
            return self._heads[0]

    @head.setter
    def head(self, head: BaseHead):
        if not isinstance(head, BaseHead):
            raise TypeError(f'Invalid type for "heads". {type(head)}')

        self.heads = [head]

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f'Define {self.__class__.__name__}.encode()')

    def decode_train(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self.decode_eval(inputs)

    def decode_eval(self, inputs: dict[str, Any]) -> dict[str, Any]:
        NotImplementedError(f'Define {self.__class__.__name__}.decode()')

    def decode(self, inputs):
        if self.training:
            return self.decode_train(inputs)
        else:
            return self.decode_eval(inputs)

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        encoder_outputs = self.encode(inputs)
        decoder_inputs = merge_dicts([inputs, encoder_outputs])
        outputs = self.decode(decoder_inputs)

        return outputs

    def make_chunk_specs(self):
        pass

    def calculate_loss(
        self, outputs: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, Any]:
        loss = 0
        for head in self.heads:
            loss += head.loss_weight * head.calculate_loss(outputs, batch)

        return loss

    def forward_metrics(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        metrics = {}
        for head in self.heads:
            if not head.has_metrics:
                continue
            # It's a dictionary's update method.
            # Don't confuse with update of TorchMetric.
            metrics.update(
                # __call__ of TorchMetric.
                head.metrics(outputs=outputs, batch=batch, stage=stage)
            )

        return metrics

    def update_metrics(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        stage: str,
    ) -> None:
        for head in self.heads:
            if not head.has_metrics:
                continue
            head.metrics.update(outputs=outputs, batch=batch, stage=stage)

    def compute_metrics(self, stage: str) -> dict[str, Any]:
        metrics = {}
        for head in self.heads:
            if not head.has_metrics:
                continue
            metrics.update(head.metrics.compute(stage=stage))

        return metrics

    def reset_metrics(self, stage: str) -> None:
        for head in self.heads:
            if not head.has_metrics:
                continue
            head.metrics.reset(stage=stage)

    def training_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> dict[str, Any]:
        outputs = self(batch)
        loss = self.calculate_loss(outputs, batch)

        # Update and evaluate metric.
        metrics = self.forward_metrics(outputs, batch, stage='train')

        self.log('train/loss', loss)
        # Log instant metrics.
        self.log_dict(metrics)

        return loss

    def on_train_epoch_end(self) -> None:
        # Don't log epoch averaged metrics. Just reset the states.
        self.reset_metrics(stage='train')

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        outputs = self(batch)
        loss = self.calculate_loss(outputs, batch)

        # Don't log metrics yet.
        self.update_metrics(outputs, batch, stage='val')

        # loss will be epoch averaged.
        self.log('val/loss', loss)

    def on_validation_epoch_end(self) -> None:
        metrics = self.compute_metrics(stage='val')
        self.log_dict(metrics)
        self.reset_metrics(stage='val')

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        outputs = self(batch)
        loss = self.calculate_loss(outputs, batch)

        # Don't log metrics yet.
        self.update_metrics(outputs, batch, stage='test')

        # loss will be epoch averaged.
        self.log('test/loss', loss)

    def on_test_epoch_end(self) -> None:
        metrics = self.compute_metrics(stage='test')
        self.log_dict(metrics)
        self.reset_metrics(stage='test')
