'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 48, 5)
        self.bn1   = nn.BatchNorm2d(48) 
        self.prelu1 = nn.PReLu()

        self.conv2 = nn.Conv2d(48, 16, 5)
        self.bn2   = nn.BatchNorm2d(16)
        self.prelu2 = nn.PReLu()

        self.fc1   = nn.Linear(16*5*5, 10)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
