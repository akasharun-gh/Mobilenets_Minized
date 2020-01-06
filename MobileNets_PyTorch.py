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

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
    
    images = images[0:4]
    labels = labels[0:4]

    # print labels
    print(' '.join('%s' % classes[labels[j]] for j in range(4)))
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

        nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
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
        self.conv1 = nn.Conv2d(3, 50, 3)
        # Input is 32x32x3
        # Output is 30x30x50
        
        res1,tot_conv = total_conv_mults(3,3,50,30)
        print("\nConv_layer : Multiplications of layer 1 =  ",res1)

        self.conv2 = conv_dw(50, 60, 1)
        # Input is 30x30x50
        # Output is 14x14x60 because os 2x2 pooling
        
        res,tot = total_dw_mults(50,3,60,28)
        res_conv,tot_conv = total_conv_mults(50,3,60,28)

        print("Depth_wise : Multiplications of layer 2 = ",res)


        self.conv3 = conv_dw(60, 72, 1)
        # Input is 14x14x60
        # Output is 12x12x72
        
        res,tot = total_dw_mults(60,3,72,12)
        res_conv,tot_conv = total_conv_mults(60,3,72,12)

        print("Depth_wise : Multiplications of layer 3 = ",res)

        
        self.conv4 = conv_dw(72, 80, 1)
        # Input is 12x12x72
        # Output is 10x10x80
        
        res,tot = total_dw_mults(72,3,80,10)
        res_conv,tot_conv = total_conv_mults(72,3,80,10)

        print("Depth_wise : Multiplications of layer 4 = ",res)

        print("\nTotal multiplications of all conv+depth_wise layers",res1+tot) 
        print("\nTotal multiplications if only conv layers",tot_conv,"\n")

        
        self.fc1 = nn.Linear(80*10*10, 10)
        

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)             # 2x2 max pooling, stride=2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 80*10*10)   
        x = F.log_softmax(self.fc1(x), dim=1) 
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
print(myNet)
print("Total number of parameters: ", myNet.num_params())
 
###################################
# Training

import torch.optim as optim

# Loss function: negative log likelihood
criterion = nn.NLLLoss()

# Configuring stochastic gradient descent optimizer
optimizer = optim.SGD(myNet.parameters(), lr=learning_rate, momentum=momentum)

# Each epoch will go over training set once; run two epochs
for epoch in range(epochs): 

    running_loss = 0.0

    # iterate over the training set
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data

        # Clear the parameter gradients
        optimizer.zero_grad()

        #################################
        # forward + backward + optimize

        # 1. evaluate the current network on a minibatch of the training set
        outputs = myNet(inputs)              

        # 2. compute the loss function
        loss = criterion(outputs, labels)  

        # 3. compute the gradients
        loss.backward()                    

        # 4. update the parameters based on gradients
        optimizer.step()                   

        # Update the average loss
        running_loss += loss.item()

        # Print the average loss every 256 minibatches ( == 16384 images)
        if i % 256 == 255:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 256))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():       # this tells PyTorch that we don't need to keep track
                                # of the gradients because we aren't training
        for data in testloader:
            images, labels = data
            outputs = myNet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Epoch %d: Accuracy of the network on the %d test images: %d/%d = %f %%' % (epoch+1, total, correct, total, (100 * correct / total)))


print('Finished Training!')



###################################
# Let's look at some test images and see what our trained network predicts for them

if (showImages > 0):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images_f = images[0:4]
    labels = labels[0:4]
    outputs = myNet(images_f)
    im_data1 = images[0]
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', ' '.join('%10s' % classes[predicted[j]] for j in range(4)))

    imshow(torchvision.utils.make_grid(images_f))
    '''img_f = open("image_data.txt", "w")
    for pix in im_data1:
        img_f.write(str(pix.hex()) + '\n')
    img_f.close'''


##################################
# Let's comptue the total accuracy across the training set

correct = 0
total = 0
with torch.no_grad():       # this tells PyTorch that we don't need to keep track
                            # of the gradients because we aren't training
    for data in trainloader:
        images, labels = data
        outputs = myNet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d training images: %f %%' % (total, (100 * correct / total)))



##################################
# Now we want to compute the total accuracy across the test set

correct = 0
total = 0
with torch.no_grad():       # this tells PyTorch that we don't need to keep track
                            # of the gradients because we aren't training
    for data in testloader:
        images, labels = data
        outputs = myNet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d/%d = %f %%' % (total, correct, total, (100 * correct / total)))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = myNet(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %10s : %f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
 
'''f = open("dw_weights_data.txt", "w")    
trained_weights = myNet.conv1.weight.data   
for wts in trained_weights:
    f.write(str(wts.hex()) + '\n')
    
f.close()

f2 = open("pw_weights_data.txt", "w")    
pw_weights = myNet.conv2.weight.data   
for pw_wts in pw_weights:
    f2.write( str(pw_wts.hex()) + '\n')
    
f2.close()'''

torch.save(myNet.state_dict(), 'mobnets_trained.pt')
torch.save(myNet, 'model_mobnets.pt')

    
    


