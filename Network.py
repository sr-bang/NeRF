import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def linear_block(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())
# --> nn.Linear(3 + 3 * 2 * dim_encode, filter_size)

class model_NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = linear_block(39,256)
        self.l2 = linear_block(256,256)
        self.l3 = linear_block(256+39,256)
        self.l4 = linear_block(256,128)
        self.l5 = nn.Linear(128,4)
        # self.double()
 

    def forward(self, gamma):
        temp = gamma
        x = self.l1(gamma)  #39-256
        x = self.l2(x)      #256-256
        x = self.l2(x)      #256-256
        x = self.l2(x)      #256-256
        x = self.l3(torch.concat([x , temp], axis=-1))      #256+39-256
        x = self.l2(x)      #256-256
        x = self.l2(x)      #256-256
        x = self.l4(x)      #256-128 
        x = self.l5(x)      #128-4
        return x
    

# For tiny NeRF:

class model_tinyNeRF(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = linear_block(39,128)
    self.l2 = linear_block(128,128)
    self.l3 = nn.Linear(128,4)


  def forward(self, gamma):
    x = self.l1(gamma)
    x = self.l2(x)
    x = self.l3(x)
    return x
