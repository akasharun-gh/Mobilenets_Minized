# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:53:23 2019

@author: akash
"""
batchSize = 8           # The batch size used for learning

import torch
import numpy as np
import torchvision
import torch.nn as nn
import bitstring
#np.set_printoptions(formatter={'float':lambda x:float(x).hex()})
#device = torch.device('cpu')
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

'''class Net(nn.Module):
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

        return numParams'''
    
model = torch.load('model_mobnets_mod.pt')
x = model.conv1[0].weight.data.numpy()

dataiter = iter(trainloader)
images, labels = dataiter.next()
#images_f = images[0:4]
#labels = labels[0:4]
im_data1 = images[0:2]

img_f = open("image_data_mod.txt", "w")
countx = 0
for imgse in im_data1:
    for pix in imgse:
        for pts in pix:
            for its in pts:
                #print(its.item())
                countx = countx+1
                img_f.write(str(bitstring.BitArray(float=(its.item()), length=32)) + '\n')
img_f.close()
print("countx=",countx)

count = 0
f = open("dw_weights_data_mod.txt", "w") 
for i in x:
    for t in i:
        for c in t:
            for a in c:
                #print((a.item()).hex())
                f.write(str(bitstring.BitArray(float=(a.item()), length=32)) + '\n')
                count = count+1
            
print("count=",count)              
f.close()
#print("x=", x)

count2 = 0
f2 = open("pw_weights_data_mod.txt", "w")    
pw_weights = model.conv1[3].weight.data   
for pw_wts in pw_weights:
    for p in pw_wts:
        for w in p:
            for t in w:
                #print(bitstring.BitArray(float=(t.item()), length=32))
                f2.write( str(bitstring.BitArray(float=(t.item()), length=32)) + '\n')
                count2 = count2 +1
                 
    
print("count2=",count2) 
f2.close()

