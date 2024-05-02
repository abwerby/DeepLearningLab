import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3):
        super(CNN, self).__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(history_length+1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Fully-connected layers
        self.fc1 = nn.Linear(64*21*21, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


class DeepCNN(nn.Module):

    def __init__(self, history_length=0, n_classes=5):
        super(DeepCNN, self).__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(history_length+1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Fully-connected layers
        self.fc1 = nn.Linear(256*6*6, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)
        return x