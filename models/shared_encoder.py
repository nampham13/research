import torch.nn as nn
from torchvision.models import resnet50

class SharedEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet50(weights=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  

    def forward(self, x):
        return self.features(x)

