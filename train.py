import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets

import transforms
from capsnet import CapsNet


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
        # transforms.RandomShift(2),
        transforms.ToTensor()
    ])), shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])))

model = CapsNet()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
model.train()

for epoch, (input, target) in enumerate(train_loader, 1):
    input = Variable(input)
    target = Variable(target)

    output = model(input)
    criterion = torch.nn.MultiMarginLoss(2, 0.9, size_average=False)
    loss = criterion(output.norm(dim=1), target)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(loss)
