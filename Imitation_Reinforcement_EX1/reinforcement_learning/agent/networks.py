import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

# model for CartPole
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# model for CarRacing
class CNN(nn.Module):

    def __init__(self, history_length=0, action_dim=5):
        super(CNN, self).__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(history_length+1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Fully-connected layers
        self.fc1 = nn.Linear(64*24*24, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, action_dim)
    
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

    