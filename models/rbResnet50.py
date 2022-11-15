__all__ = ["RBResNet50"]

import torch.nn as nn
from torchvision.models import resnet50


class RBResNet50(nn.Module):
    """
    A wrapper around resnet50 for binary classification based
    in 3 input images: reference, science, difference, in that order.

    The original resnet50 is introduced in the below publication:
    doi.org/10.48550/arXiv.1512.03385
    """

    def __init__(self):
        super().__init__()

        self.resnet = resnet50(pretrained=False)

        # Change the number of input channels to 3
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Change the number of connections in the last fc layer
        # according to the input size (3x256x256).
        # IMPORTANT: needs to be updated if the standard input
        # size changes.
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1, bias=True),
        )

    def forward(self, x):
        return self.resnet(x)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
