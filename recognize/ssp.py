# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 23:39:55 2017

@author: misakawa
"""

import torch
import numpy as np
from linq import Flow
from functools import reduce
F = torch.nn.functional 
D = torch.autograd.grad
Rand = torch.rand


def and_then(*fs):
    def call(x):
        return reduce(lambda a, b: b(a), fs, x)
    return call

def Var(v):
    return torch.autograd.Variable(v, requires_grad=True)

def Val(v):
    return torch.autograd.Variable(v, requires_grad=False)

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1) 
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)  
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x):
        out = and_then(self.fc1, 
                       self.relu1,
                       self.fc2,
                       self.relu2,
                       self.fc3)(x)
        return out

class SSP(torch.nn.Module):
    
    def __init__(self, MLP: torch.nn.Module):
        super(SSP, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.disc1 = torch.nn.Dropout(0.3)
        
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2)
        self.disc2 = torch.nn.Dropout(0.3)
        
        self.conv3 = torch.nn.Conv2d(32, 64, 5)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2)
        self.disc3 = torch.nn.Dropout(0.3)
        
        self.sp1   = torch.nn.AdaptiveAvgPool2d(4)
        self.sp2   = torch.nn.AdaptiveAvgPool2d(8)
        self.flatten = lambda v: v.view(v.shape[0], -1)
        self.predictor = MLP    
    
    def forward(self, x):
        spl = and_then(
                self.conv1,
                self.pool1,
                F.relu,
                self.disc1,
                
                self.conv2,
                self.pool2,
                F.relu,
                self.disc2,
                
                self.conv3,
                self.pool3,
                F.relu,
                self.disc3)(x)
        
        sp_func = lambda x: and_then(x, self.flatten)(spl)
        out1, out2 = sp_func(self.sp1), sp_func(self.sp2)
        catted = torch.cat([out1, out2], dim=1)
        out = self.predictor(catted)
        return out
    def fit(self, samples, labels, 
            lr=3e-4, num_epochs=100):
        samples = [np.reshape(sample, newshape=(1, *shape)) for sample,shape in map(lambda x: (x, x.shape), samples)]
        labels  = np.reshape(labels, newshape=(labels.shape[0], 1, labels.shape[1]))
        labels  = Val(torch.from_numpy(labels).long())
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) 
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            loss = 0
            optimizer.zero_grad()
            for (x, y) in zip(samples, labels):
                x = Val(torch.from_numpy(x).float())
                outputs  = self(x)
#                print(outputs.shape, y.shape)
                loss = loss + criterion(outputs, y[0]) # bug
            loss.backward()
            optimizer.step()
            print('Epoch [%d/%d], Loss: %.4f'%(epoch+1, num_epochs, loss.data[0]))
    
    def predict(self, samples):
        samples = [np.reshape(sample, newshape=(1, *shape)) for sample,shape in map(lambda x: (x, x.shape), samples)]
        return np.argmax(
                [outs[0].data.numpy().flatten() for outs in 
                 map(and_then(torch.from_numpy, 
                              lambda x: x.float(),
                              Val, 
                              self), 
                    samples)], 1)

try:
    from .utils import dump_model, load_model, read_directory
except:
    from utils import dump_model, load_model, read_directory


def train(new=False, md_name='mml', epoch=100):
    import os
    dir = os.path.split(__file__)[0]
    try:
        if new:
            raise 
        ssp = load_model(f'{dir}/{md_name}')
    except:
        print('redef')
        ssp  = SSP(Net(5120, 3000, 1000, 2))

    X_pos, y_pos = read_directory(f'{dir}/pos', 1)

    X_neg, y_neg = read_directory(f'{dir}/neg', 0)


    X = X_pos + X_neg
    y = np.vstack((y_pos, y_neg))

    ids = np.random.permutation(len(X))
    X = [X[i] for i in ids]
    y = y[ids]
    #
    ssp.fit(X, y, num_epochs=epoch)

    dump_model(ssp, f'{dir}/{md_name}')

    from sklearn.metrics.classification import classification_report, confusion_matrix

    pred = ssp.predict(X)
    print(classification_report(y.flatten(), pred))
    print(confusion_matrix(y.flatten(), pred))
        
        
        
    

