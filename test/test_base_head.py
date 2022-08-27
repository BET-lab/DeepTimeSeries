import sys
sys.path.append('..')
import pytest

import torch.nn as nn
from deep_time_series.model import BaseHead


def test_tag_assignment():
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


def test_loss_weight_assignment():
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