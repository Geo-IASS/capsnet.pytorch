import torch.nn as nn
import torch.nn.functional as F

from routing import Routing


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 9)
        self.primary_caps = nn.Conv2d(256, 32, 9, stride=2)
        self.digit_caps = Routing(4 * 6 * 6, 10, 8, 16, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        return x


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    net = CapsNet()
    x = torch.rand(1, 1, 28, 28)
    net(Variable(x))
