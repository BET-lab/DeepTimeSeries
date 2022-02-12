import torch
import torch.nn as nn


class DictLoss(nn.Module):
    def __init__(self, loss_dict, weight_dict=None):
        super().__init__()
        # self.loss_dict = nn.ModuleDict(loss_dict)
        self.loss_dict = loss_dict
        self.weight_dict = weight_dict

    def forward(self, outputs, batch):
        loss = 0
        for tag, loss_fn in self.loss_dict.items():
            value = loss_fn(outputs[tag], batch[tag])
            if self.weight_dict:
                value = self.weight_dict[tag] * value
            loss += value
        return loss