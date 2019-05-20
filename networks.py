# imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet5
class LeNet5(nn.Module):
    """
    The LeNet5 neural network architecture for a 3 channel 32x32
    input as expected from the CIFAR10 dataset
    """

    def __init__(self, bias=True):
        "setup the neural network"
        
        # initalise
        super(LeNet5, self).__init__()

        # general params
        kernal = 5
        stride = 1

        # create each layer
        self.C1_layer = nn.Conv2d(1, 6, padding=2, kernel_size=kernal, stride=stride, bias=bias)
        self.C3_layer = nn.Conv2d(6, 16, padding=0, kernel_size=kernal, stride=stride, bias=bias)
        
        self.S2_layer = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.S4_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.F5_layer = nn.Linear(16*5*5, 120, bias=bias)
        self.F6_layer = nn.Linear(120,84, bias=bias)
        
        self.output = nn.Linear(84, 10, bias=bias)
        
        # create the activation function
        act = nn.ReLU()
        
        # create a list of fucntion order
        self.layers = [self.C1_layer, act, 
                       self.S2_layer, act, 
                       self.C3_layer, act, 
                       self.S4_layer, act, self.flatten, 
                       self.F5_layer, act,
                       self.F6_layer, act,
                       self.output]
        
    def flatten(self, T):
        "flatten the image for the fully connected layers"
        return T.view(-1, 16*5*5)
        
        
    def forward(self, x):
        "Pass through the neural network"
        for f in self.layers:
            x = f(x)
        return x
