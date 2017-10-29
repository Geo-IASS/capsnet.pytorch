import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import squash


class Routing(nn.Module):
    def __init__(self, num_in_caps, num_out_caps, in_dim, out_dim, num_shared):
        super(Routing, self).__init__()
        self.in_dim = in_dim
        self.num_shared = num_shared

        self.W = [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_shared)]
        self.b = Variable(torch.zeros(num_in_caps, num_out_caps))

    def forward(self, input):
        # TODO: make it work for batch sizes > 1
        _, in_channels, h, w = input.size()
        assert in_channels == self.num_shared * self.in_dim

        input = input.squeeze().view(self.num_shared, -1, self.in_dim)
        groups = input.chunk(self.num_shared)
        u = [group.squeeze().chunk(h * w) for group in groups]
        pred = [self.W[i](in_vec.squeeze()) for i, group in enumerate(u) for in_vec in group]
        pred = torch.stack([torch.stack(p) for p in pred]).view(self.num_shared * h * w, -1)

        c = F.softmax(self.b)
        s = torch.matmul(c.t(), pred)
        v = squash(s)
        self.b = torch.add(self.b, torch.matmul(pred, v.t()))
        return v


if __name__ == '__main__':
    l = Routing(4 * 6 * 6, 10, 8, 16, 4)
    print(l(Variable(torch.rand(1, 32, 6, 6))))
