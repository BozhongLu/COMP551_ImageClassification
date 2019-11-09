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

"""Variables to determine"""
batch_size = 64
device = torch.device("cpu")


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
        self.fc1 = nn.Linear(320,60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self,x):
        # Pictures into layer1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # classifier, so all need to be 0 or 1
        return F.log_softmax(x, dim=1)


model = TheNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

def train(epoch):
    model.train()
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

"""
for epoch in range(1,20):
    train(epoch)
    test ()
"""
# 99% accuracy after 11 epochs

#torch.save(model.state_dict(), "C:/Users/User/Documents/2_Programming/Machine_Learning/COMP 551/Project3/model_e20_A99")

model = TheNet()

model.load_state_dict(torch.load("C:/Users/User/Documents/2_Programming/Machine_Learning/COMP 551/Project3/model_e20_A99"))
model.eval()


test_images = pd.read_pickle('test_max_x')
testPreprocessed = np.zeros([3, 50000, 28, 28])
testPreprocessed = imagePreprocessing(test_images)
results = np.zeros([4, 50000])

from PIL import image

for ex in range(0,len(testPreprocessed[1])):
    for nr in range (0,3):
        img= torch.tensor(testPreprocessed[nr][ex])
        img = img.to(device)
        img = Variable(img)
        img = img[None, None]
        img = img.type('torch.FloatTensor') # instead of DoubleTensor
        out = model(img)
        pred = out.data.max(1,keepdim=True)[1]
        results[nr][ex]=np.int(pred[0][0])
        for t in range(0,4):
            transposed = colorImage.transpose(Image.ROTATE_90)



    if ex%100 ==0:
        print(str(ex) + "   finished")

results[3] = np.amax(results[0:3], axis=0)

# Compare the result and prediction
train_labels = pd.read_csv('train_max_y.csv')
train_labels=train_labels.iloc[:,1]
prediction=np.transpose(results[3])


import sklearn.metrics
sklearn.metrics.accuracy_score(train_labels,prediction)
sklearn.metrics.confusion_matrix(train_labels,prediction)

