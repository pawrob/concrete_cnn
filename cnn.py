import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 'valid')
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 3)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 'valid')
        self.fc1 = nn.Linear(43264, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = self.pool(x)
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
