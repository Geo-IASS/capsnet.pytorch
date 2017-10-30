import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets

import transforms
from capsnet import CapsNet
from loss import DigitMarginLoss


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
criterion = DigitMarginLoss()
model.train()

for epoch in range(1, 11):
    epoch_tot_loss = 0
    for batch, (input, target) in enumerate(train_loader, 1):
        input = Variable(input)
        target = Variable(target)
    
        output = model(input)
        loss = criterion(output, target)
        epoch_tot_loss += loss
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        print(loss.data[0], (epoch_tot_loss / batch).data[0])
