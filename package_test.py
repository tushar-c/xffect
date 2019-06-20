'''
script for checking various package functionalities crazily

#TODO: make proper testing modules

'''

import mne.decoding 
import xffect 
import mne
import numpy as np 
from xffect.processing import make_filter, pca, ica
from xffect.communication import send_data, receive_data
from xffect.training import logistic_regression, linear_regression, ridge_regression
from xffect.training import lda, svm, qda, cross_validation
from xffect.training import train, test
import torch 
import torch.nn as nn 
import torch.optim as optim
from xffect.datasets import emotion_loader

info = mne.create_info(2, 100.)
event_id = dict(left=0, right=1)
A = np.random.randn(10, 2, 40)
b = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

events = np.array([[i, 0, b[i]] for i in range(10)])
epochs = mne.EpochsArray(A, info=info, events=events, event_id=event_id)

print(epochs.get_data().shape)

transform=True

fitted = pca(epochs, info, transform=transform)

if transform:
    print('pca_fit:', fitted.get_data().shape)


ica_fit = ica(epochs, info, transform=transform)

if transform:
    print('ica fit:', ica_fit.get_data().shape)



print(cross_validation(linear_regression, epochs))
print(cross_validation(logistic_regression, epochs))
print(cross_validation(svm, epochs))
print(cross_validation(ridge_regression, epochs))
print(cross_validation(lda, epochs))
print(cross_validation(qda, epochs))

print(linear_regression(epochs))
print(logistic_regression(epochs))
print(svm(epochs))
print(ridge_regression(epochs))
print(lda(epochs))
print(qda(epochs))


class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.forward1 = nn.Linear(20, 40)
        self.forward2 = nn.Linear(40, 30)
        self.forward3 = nn.Linear(30, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        x = self.relu(self.forward1(x))
        x = self.relu(self.forward2(x))
        x = self.softmax(self.forward3(x))
        return x 


def prep():    
    features = torch.tensor([np.random.randn(1, 20) for _ in range(2000)])
    labels = [[0] for _ in range(1000)]
    labels.extend([[1] for _ in range(1000)])

    labels = torch.tensor(labels)

    net = FFNet()

    EPOCHS = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    return features, labels, net, criterion, optimizer, EPOCHS

import time 

features, labels, net, criterion, optimizer, EPOCHS = prep()
start = time.time()
train(features, labels, net, criterion, optimizer, EPOCHS, optim_every=5)
end = time.time() - start 
print('training took {} seconds'.format(end))
print(test(features, labels, net, criterion))

data = emotion_loader.get_file()
print(data)

