import torch
import torch.nn as nn
from config import NUM_CLASSES, USE_CONV_DROPOUT, DROPOUT_CONV_P, DROPOUT_FC_P


def conv_dropout_layer():
    if USE_CONV_DROPOUT and DROPOUT_CONV_P > 0:
        return nn.Dropout2d(p=DROPOUT_CONV_P)
    return nn.Identity()


class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dropout_layer(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dropout_layer(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dropout_layer(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_FC_P),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def get_model(device):
    return BaselineCNN().to(device)