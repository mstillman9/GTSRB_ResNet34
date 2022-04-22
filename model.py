import os
import json
import time
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.resnet1 = models.resnet34(pretrained=False, progress=True)
        self.resnet = nn.Sequential(*list(self.resnet1.children())[:-1])
        self.linear = nn.Linear(512, 43)        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 512)
        x = self.linear(x)        
        return x
