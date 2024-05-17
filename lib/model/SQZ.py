import torch
import torch.nn as nn


class SQZ_Module(nn.Module):
    def __init__(self):
        super(SQZ_Module, self).__init__()
        
    def forward(self, feat):
        tmp = []
        for i in feat:
            tmp.append(i.squeeze(2))
        return tmp