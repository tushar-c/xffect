'''
module for training neural networks
'''

import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np

#TODO: replace each sample fetching with DataLoader


def train(features, labels, net_object, criterion, optimizer, EPOCHS,
            log=True, optim_every='sample'):

    '''
    function for training a neural network

    Parameters: features -> numpy array or torch tensor containing training features
                labels -> numpy array or torch tensor containing training labels
                net_object -> torch neural network object which to train
                criterion -> torch loss function object, to which minimize
                optimizer -> torch object responsible for optimization
                EPOCHS -> int object that denotes number of passes through th dataset
                log -> whether to log information or not
                optim_every -> string denoting when to run the optimizer object
    '''

    allowed_update_stages = ('epoch', 'sample')

    if isinstance(optim_every, (int, float)):
        optim_every = int(optim_every)
    else:
        if optim_every not in allowed_update_stages:
            raise Exception('optim_every must be in {}'.format(optim_every), 
                            'or be either an int or a float')

    if not isinstance(features, (list, torch.Tensor, np.ndarray)):
        raise Exception('features must be either a list, torch.Tensor or' 
                        'ndarray')
    
    if len(features) < 1:
        raise Exception('features cannot be empty')

    if not isinstance(labels, (list, torch.Tensor, np.ndarray)):
        raise Exception('features must be either a list, torch.Tensor or' 
                        'ndarray')

    if len(labels) < 1:
        raise Exception('labels cannot be empty')

    if not hasattr(optimizer, 'step'):
        raise Exception('optimizer must be a valid torch.optim optimizer')

    EPOCHS = int(EPOCHS)

    N = len(features)

    for e in range(EPOCHS):
        epoch_loss = 0
        epoch_accuracy = 0
        for n in range(N):
            feature, label = features[n].float(), labels[n].long()
            
            forward = net_object.forward(feature)
            loss = criterion(forward, label)

            epoch_loss += loss.item()
            loss.backward(retain_graph=True)

            if optim_every == 'sample':
                optimizer.step()
                optimizer.zero_grad()

            if isinstance(optim_every, int):
                if n % optim_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if torch.argmax(forward).item() == label.item():
                epoch_accuracy += 1
        
        if optim_every == 'epoch':
            optimizer.step()
            optimizer.zero_grad()

        if log:
            print('Epoch {} / {}; Loss {}; Accuracy {}'.format(e + 1, EPOCHS, epoch_loss / N, epoch_accuracy / N))


def test(features, labels, net_object, criterion):

    '''
    function for testing a neural network

    Parameters: features -> numpy array or torch tensor containing test features
                labels -> numpy array or torch tensor containing test labels
                net_object -> torch neural network object which is to be tested
                criterion -> torch object representing the loss function

    '''
    if not isinstance(features, (list, torch.Tensor, np.ndarray)):
        raise Exception('features must be either a list, torch.Tensor or' 
                        'ndarray')

    if not isinstance(labels, (list, torch.Tensor, np.ndarray)):
        raise Exception('features must be either a list, torch.Tensor or' 
                        'ndarray')

    if len(features) < 1:
        raise Exception('features cannot be empty')

    if len(labels) < 1:
        raise Exception('labels cannot be empty')

    test_loss = 0
    test_score = 0
    N = len(features)
    
    with torch.no_grad():
        for n in range(N):
            feature, label = features[n].float(), labels[n].long()
            forward = net_object.forward(feature)
                
            loss = criterion(forward, label)
            test_loss += loss.item()

            if torch.argmax(forward).item() == label.item():
                test_score += 1
            
    return test_loss / N, test_score / N

