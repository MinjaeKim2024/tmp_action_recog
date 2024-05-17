'''
This file is modified from:
https://github.com/zhoubenjia/RAAR3DNet/blob/master/Network_Train/lib/model/RAAR3DNet.py
'''

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
from torchvision.utils import save_image, make_grid

def tensor_split(t):
    split_result = [t[:, :, i, :, :] for i in range(64)]
    # arr = torch.split(t, 1, dim=2)
    # arr = [x.squeeze(2) for x in arr]
    return split_result
def tensor_merge(arr):
    arr = [x.unsqueeze(1) for x in arr]
    t = torch.cat(arr, dim=1)
    return t.permute(0, 2, 1, 3, 4)

class FRP_Module(nn.Module):
    def __init__(self, w, inplanes):
        super(FRP_Module, self).__init__()
        self._w = w
        self.rpconv1d = nn.Conv1d(2, 1, 1, bias=False)  # Rank Pooling Conv1d, Kernel Size 2x1x1
        self.rpconv1d.weight.data = torch.FloatTensor([[[1.0], [0.0]]])
        # self.bnrp = nn.BatchNorm3d(inplanes)  # BatchNorm Rank Pooling
        # self.relu = nn.ReLU(inplace=True)
        self.hapooling = nn.MaxPool2d(kernel_size=2)
        self.repeat_dic = {28:3, 14:4, 7:5}
        self.size_map = torch.tensor([28, 14, 7])
        self.repeat_map = torch.tensor([3, 4, 5])
        
        
    def forward(self, x, datt=None):
        def run_layer_on_arr(arr, l):
            return [l(x) for x in arr]
        def oneconv(a, b):
            s = a.size()
            c = torch.cat([a.contiguous().view(s[0], -1, 1), b.contiguous().view(s[0], -1, 1)], dim=2)
            c = self.rpconv1d(c.permute(0, 2, 1)).permute(0, 2, 1)
            return c.view(s)
        tarr = tensor_split(x)
        garr = tensor_split(datt)
        # init part define
        size_idx = (self.size_map == tarr[0].size()[3]).nonzero(as_tuple=True)[0]
        rep = self.repeat_map[size_idx]
        for _ in range(rep):
            garr = run_layer_on_arr(garr, self.hapooling)
        
        
        
        attarr = [a * (b + torch.ones(a.size()).cuda()) for a, b in zip(tarr, garr)]
        datt = [oneconv(a, b) for a, b in zip(tarr, attarr)]
        return tensor_merge(datt)
        

if __name__ == '__main__':
    model = SATT_Module().cuda()
    inp = torch.randn(2, 3, 64, 224, 224).cuda()
    out = model(inp)
    print(out.shape)
