#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (t-sigai at microsoft dot com)                 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, densenet121
#from torchsummary import summary

class model(nn.Module):
    def __init__(self, num_classes=10, drop_prob=0.5):
        super(model, self).__init__()

        # encoder
        self.model_ft = resnet34(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features#*4
        self.model_ft.fc = nn.Sequential(nn.Dropout(drop_prob), nn.Linear(num_ftrs, 128), nn.ReLU(True), 
                nn.Dropout(drop_prob), nn.Linear(128, 128), nn.ReLU(True))
        self.cls_fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.model_ft(x)
        x = self.cls_fc(x)
        return x

    def fine_tune(self, block_layer=5):

        for idx, child in enumerate(self.model_ft.children()):
            if idx>block_layer:
                break
            for param in child.parameters():
                param.requires_grad = False
