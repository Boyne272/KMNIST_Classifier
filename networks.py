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


# AlexNet_half
# class AlexNet_half(nn.Module):
    # """
    
    # """

    # def __init__(self, bias=True):
        # "setup the neural network"
        
        # # initalise
        # super(AlexNet_half, self).__init__()

        # # Original AlexNet
        # # self.C1_layer = nn.Conv2d(3, 48, padding=0, kernel_size=11, stride=4, bias=bias)
		# # self.P1_layer = nn.MaxPool2d(kernel_size=3, stride=2)
        # # self.C2_layer = nn.Conv2d(48, 128, padding=2, kernel_size=5, stride=1, bias=bias)
        # # self.P2_layer = nn.MaxPool2d(kernel_size=3, stride=2)
		# # self.C3_layer = nn.Conv2d(128, 192, padding=2, kernel_size=3, stride=1, bias=bias)
		# # self.C4_layer = nn.Conv2d(192, 128, padding=0, kernel_size=3, stride=1, bias=bias)
        # # self.F5_layer = nn.Linear(4068, 2048, bias=bias)
        # # self.F6_layer = nn.Linear(2048, 2048, bias=bias)
        # # self.output = nn.Linear(2048, 10, bias=bias)
		
		# # reduced AlexNet
		# self.C1_layer = nn.Conv2d(1, 24, padding=1, kernel_size=3, stride=1, bias=bias)
		# self.P1_layer = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.C2_layer = nn.Conv2d(24, 64, padding=0, kernel_size=5, stride=1, bias=bias)
        # self.P2_layer = nn.MaxPool2d(kernel_size=3, stride=2)
		# self.C3_layer = nn.Conv2d(64, 96, padding=2, kernel_size=3, stride=1, bias=bias)
		# self.C4_layer = nn.Conv2d(96, 64, padding=2, kernel_size=3, stride=1, bias=bias)
        # self.F5_layer = nn.Linear(4068, 2048, bias=bias)
        # self.F6_layer = nn.Linear(2048, 2048, bias=bias)
        # self.output = nn.Linear(2048, 10, bias=bias)
        
        # # create the activation function
        # act = nn.ReLU()
        
        # # create a list of fucntion order
        # self.layers = [self.C1_layer, act, 
                       # self.P1_layer, act, 
                       # self.C2_layer, act, 
                       # self.P2_layer, act,
					   # self.C3_layer, act,
					   # self.C4_layer, act,
					   # self.C5_layer, act,
					   # self.flatten, 
                       # self.F5_layer, act,
                       # self.F6_layer, act,
                       # self.output]
        
    # def flatten(self, T):
        # "flatten the image for the fully connected layers"
		# s = T.size(1)*T.size(2)*T.size(3)
        # return T.view(-1, 16*5*5)
        
        
    # def forward(self, x):
        # "Pass through the neural network"
        # for f in self.layers:
            # x = f(x)
        # return x