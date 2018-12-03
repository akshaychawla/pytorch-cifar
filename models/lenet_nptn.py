'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from .nptn import NPTN_vanquish

class LeNet_NPTN(nn.Module):
    def __init__(self, in_chs=(3,48), out_chs=(48,16), Gs=(1,1)):
        super(LeNet_NPTN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = NPTN_vanquish(in_chs[0], out_chs[0], G=Gs[0], k=5, pad=0, stride=1)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = NPTN_vanquish(in_chs[1], out_chs[1], G=Gs[1], k=5, pad=0, stride=1)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
