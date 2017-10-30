import torch
import torch.nn as nn
from torch.autograd import Variable


class DigitMarginLoss(nn.Module):
    def __init__(self, M=0.9, m=0.1, l=0.5):
        super(DigitMarginLoss, self).__init__()
        self.M = M
        self.m = m
        self.l = l

    def forward(self, output, target):
        norm = output.norm(dim=0)
        zero = Variable(torch.zeros(1))
        losses = [torch.max(zero, self.M - norm).pow(2) if digit == target.data[0]
            else self.l * torch.max(zero, norm - self.m).pow(2)
            for digit in range(10)]
        return torch.cat(losses).sum()
