import sys
sys.path.append('..')
import pytest

import torch
import torch.nn as nn
from deep_time_series import (
    BaseHead, Head,
)


def test_base_head_tag_assignment():
    head = BaseHead()
    head.tag = 'my_tag'

    assert head.tag == 'head.my_tag'
    assert head.label_tag == 'label.my_tag'

    head.tag = 'head.my_tag'

    assert head.tag == 'head.my_tag'
    assert head.label_tag == 'label.my_tag'

    with pytest.raises(TypeError):
        head.tag = 3

    head = BaseHead()
    with pytest.raises(TypeError):
        head.tag = nn.Module()


def test_base_head_loss_weight_assignment():
    head = BaseHead()
    assert abs(head.loss_weight - 1.0) < 1e-10

    head.loss_weight = 3.0
    assert abs(head.loss_weight - 3.0) < 1e-10

    with pytest.raises(TypeError):
        head.loss_weight = 'hello'

    with pytest.raises(TypeError):
        head.loss_weight = nn.Module()

    with pytest.raises(ValueError):
        head.loss_weight = -3


def test_head():
    head = Head(
        tag='my_tag',
        output_module=nn.Linear(5, 3),
        loss_fn=nn.MSELoss(),
        loss_weight=0.3,
    )

    assert head.tag == 'head.my_tag'
    assert head.loss_weight == 0.3

    with pytest.raises(ValueError):
        head.loss_weight = -1.0

    x = torch.rand(2, 1, 5)
    y = head(x)

    assert y.shape == torch.Size([2, 1, 3])

    outputs = head.get_outputs()

    assert torch.allclose(outputs['head.my_tag'], y)

    head.reset_outputs()

    for i in range(10):
        y = head(x)

    outputs = head.get_outputs()

    assert outputs['head.my_tag'].shape == torch.Size([2, 10, 3])

    head.reset_outputs()
    with pytest.raises(RuntimeError):
        outputs = head.get_outputs()

    batch = {
        'label.my_tag': torch.zeros(size=(2, 10, 3))
    }

    head.reset_outputs()
    for i in range(10):
        y = head(x)

    outputs = head.get_outputs()

    loss = head.calculate_loss(outputs, batch)

    assert torch.allclose(loss, (outputs['head.my_tag']**2).mean())
