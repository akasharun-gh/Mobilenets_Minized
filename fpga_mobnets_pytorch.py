# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:53:38 2019

@author: akash
"""

"""
AMS-561 Final Project

@author: Sankar Rachuru
     ID: 111684629
@author: Akash Arun
     ID: 112271185

"""


############################
# Parameters you can adjust
showImages = 1          # Will show images as demonstration if = 1
batchSize = 8           # The batch size used for learning
learning_rate = 0.01    # Learning rate used in SGD
momentum = 0.5          # Momentum used in SGD
epochs = 4              # Number of epochs to train for

############################################
# Set up our training and test data
# The torchvision package gives us APIs to get data from existing datasets like MNST
# The "DataLoader" function will take care of downloading the test and training data

import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',' deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#######################################
# Let's look at a few random images from the training data

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  #undo normalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if (showImages>0):

    # Grab random images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    images = images[0:2]
    labels = labels[0:2]

    # print labels
    print(' '.join('%s' % labels[j] for j in range(2)))
    # Show images
    imshow(torchvision.utils.make_grid(images))


##############################
# This function "conv_dw" is used in mobilenets, that consists of a 
# depth wise seperable convolution layer and a pointwise convolution layer.
# Depth wise seperable layer is achieved by changing the parameters of the nn.Conv2d function.
# The groups parameter of the nn.Conv2d function is changed to be equal to number of inputs,("inp" in this case)
# whereas in a standard convolution, the default value is given for groups, which is "groups = 1".
# In pointwise convolution we set the kernel value to 1, as we have "oup" number of 1x1xinp 
# kernels that convole to give the output feature maps. These output feature maps of the point
# pointwise convolution layers are fed to the next layer as input.
##############################    
    
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 0, groups=inp, bias=True),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, groups=1,bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    
    
# Function to calculate number of multiplications in a convolution layer
    
total_mults = 0
def total_conv_mults(inp, kernel, out, features):
    global total_mults
    res = inp*out*(kernel**2)*(features**2)
    total_mults += res
    
    return res,total_mults

# Function to calculate number of multiplications in a depthwise convolution layer
    
dw_mults_total = 0
def total_dw_mults(inp, kernel, out, features):
    global dw_mults_total
    dw = (kernel**2)*inp*(features**2)
    pw = (1*1*inp)*out*(features**2)
    res = dw + pw
    dw_mults_total += res
    
    return res,dw_mults_total

##################################
# Define our network


class Net(nn.Module):
    total_mults = 0

    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 50, 3)
        # Input is 32x32x3
        # Output is 30x30x50
        
        #res1,tot_conv = total_conv_mults(3,3,50,30)
        #print("\nConv_layer : Multiplications of layer 1 =  ",res1)

        self.conv1 = conv_dw(3, 3, 1)
        # Input is 30x30x50
        # Output is 14x14x60 because os 2x2 pooling
        
        res,tot = total_dw_mults(3,3,3,30)
        #res_conv,tot_conv = total_conv_mults(50,3,60,28)

        print("Depth_wise : Multiplications of layer 2 = ",res)
        
        #self.fc1 = nn.Linear(80*10*10, 10)
        

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        #x = x.view(-1, 3*30*30)   
        #x = F.log_softmax(self.fc1(x), dim=1) 
        return x

    # Some simple code to calculate the number of parameters
    def num_params(self):
        numParams = 0
        for param in myNet.parameters():
            thisLayerParams=1
            for s in list(param.size()):
                thisLayerParams *= s
            numParams += thisLayerParams

        return numParams
    

myNet = Net()
#print(myNet)
print("Total number of parameters: ", myNet.num_params())
 
###################################
# Training

import torch.optim as optim
import bitstring

counter = 1
outputs = myNet(images)

fl = open("final_output_data_mod.txt", "w")
for nums in outputs:
    for i in nums:
        for j in i:
            for k in j:
                fl.write( str(bitstring.BitArray(float=(k.item()), length=32)) + '\n') 
                
fl.close()
# Each epoch will go over training set once; run two epochs


print('Finished Training!')



###################################
# Let's look at some test images and see what our trained network predicts for them




 


torch.save(myNet.state_dict(), 'mobnets_trained_mod.pt')
torch.save(myNet, 'model_mobnets_mod.pt')

    
    


