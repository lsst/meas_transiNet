__all__ = ["VGG6"]

import torch
import torch.nn as nn


class VGG6(nn.Module):
    def __init__(self, input_shape=(3, 51, 51), n_classes=1):
        super(VGG6, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc1 = nn.Linear(32 * (input_shape[1] // 8) * (input_shape[2] // 8), 256)
        self.fc_out = nn.Linear(256, n_classes)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc_out(x).squeeze(1)

        return x
