import numpy as np
import torch
from torch import nn
import timm

class XrayImgClassifierEfficientnet(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
    def forward(self, x):
        x = self.model(x)
        return x

class XrayImgClassifierVit(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, n_class)
    def forward(self, x):
        x = self.model(x)
        return x
