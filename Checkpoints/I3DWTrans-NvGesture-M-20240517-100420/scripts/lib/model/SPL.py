import torch
import torch.nn as nn
class SPL_Module(nn.Module):
    def __init__(self):
        super(SPL_Module, self).__init__()
        
    def forward(self, feat):
        c = torch.split(feat, 1, dim=2)
        return c