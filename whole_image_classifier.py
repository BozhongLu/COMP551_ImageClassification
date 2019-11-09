from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
#from find_numbers import *
from torchvision import datasets, transforms, models
import os
import numpy as np
from sklearn import metrics

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')
train_labels = pd.read_csv('train_max_y.csv')


"""Variables to determine"""
batch_size = 64
device = torch.device("cpu")

torch_tensor_output = torch.tensor(train_labels.values)
torch_tensor_train = torch.from_numpy(train_images)

trainloader = torch.utils.data.DataLoader(train_images, batch_size=64)
testloader = torch.utils.data.DataLoader(test_images, batch_size=64)

X=torch.Tensor(train_images)
Y=torch.Tensor(train_labels.iloc[:,1])


def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(),])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),])
    train_data = datasets.ImageFolder(train_images, transform=train_transforms)
    test_data = datasets.ImageFolder(test_images[1],transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader,trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)


kwargs={}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

class TheNet(nn.Module):
    def __init__(self):
        super(TheNet,self).__init__()
        # convolution layer for image processing
        # put 4 pixels together to kernels
        # take 1 image in anfd 10 out
        # kernel size 5: 5x5 pixel
        self.conv1 = nn.Conv2d(1,20,kernel_size=4)
        self.conv2 = nn.Conv2d(20,20,kernel_size=4)
        # useless pixels drop out
        self.conv_dropout = nn.Dropout2d()
        # Fullyconnected layer= normal linear layers
        self.fc1 = nn.Linear(5120,4)
        self.fc2 = nn.Linear(4, 10)

    def forward(self,x):
        # Pictures into layer1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        #x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # classifier, so all need to be 0 or 1
        return F.log_softmax(x, dim=1)


model = TheNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

def train(epoch):
    #model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            test_loss += F.nll_loss(out, target, size_average=False).item() # sum up batch loss
            pred = out.data.max(1,keepdim=True)[1]
            #pred = out.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




for epoch in range(1,20):
    train(epoch)
    test ()

# 99% accuracy after 11 epochs

#torch.save(model.state_dict(), "C:/Users/User/Documents/2_Programming/Machine_Learning/COMP 551/Project3/model_e20_A99")
