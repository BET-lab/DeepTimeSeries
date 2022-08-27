import sys
sys.path.append('..')
import pytest

import torch.nn as nn
from deep_time_series.model import BaseHead


def test_tag_assignment_1():
    head = BaseHead()
    head.tag = 'my_tag'

    assert head.tag == 'head.my_tag'
    assert head.label_tag == 'label.my_tag'


def test_tag_assignment_2():
    head = BaseHead()
    head.tag = 'head.my_tag'

    assert head.tag == 'head.my_tag'
    assert head.label_tag == 'label.my_tag'


def test_tag_assignment_3():
    head = BaseHead()
    with pytest.raises(TypeError):
        head.tag = 3


def test_tag_assignment_4():
    head = BaseHead()
    with pytest.raises(TypeError):
        head.tag = nn.Module()