import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import OrderedDict

# create parse and terminal arguments
parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument("data_dir", help="Prints the supplied argument.",default ="flowers")
parser.add_argument("--save_dir", default ="checkpoint.pth")
parser.add_argument("--arch", default="densenet121")
parser.add_argument("--learning_rate", type = float, default = 0.001)
parser.add_argument("--hidden_units", type = int, default = 600)
parser.add_argument("--epochs", type = int, default = 10)
parser.add_argument("--gpu", default = 'cuda')
args = parser.parse_args()


# store terminal argument valeus 
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid' # path to valid images 
filepath = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.gpu


# training data transform
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

# test and valid data transform
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# load train data
image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)

# read train data and split it to batches 
# every batch has 64 image
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size= 64, shuffle=True)






# Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)


valid_image_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)

validloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64)

nclasses = len(image_datasets.class_to_idx)

# Use GPU if it's available
if device == 'cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# check which model is passed and change input value depend on it

# if architure is vgg16 it change input features to be 25088
if 'vgg' in arch:
    model =  models.vgg16(pretrained=True)
    ninput = 25088

    
# if architure is densenet121 change inout features to 1024
elif 'densenet121' == arch:
    model = models.densenet121(pretrained=True)
    ninput = 1024

for param in model.parameters():
    param.requires_grad = False

# define classifiers to model
model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(ninput, hidden_units)),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, nclasses)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# creat optimizer and critertion
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)



print_every = 10
steps = 0

# use gpu
model.to(device)


for e in range(epochs):
    running_loss = 0
    for inputs, labels in dataloaders:
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # edit weights
        optimizer.step()
        
        # calculate total losses
        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            valloss = 0
            accuracy=0


            for inputs, labels in validloaders:
                    optimizer.zero_grad()

                    inputs, labels = inputs.to(device) , labels.to(device)

                    with torch.no_grad():
                        outputs = model.forward(inputs)
                        valloss = criterion(outputs,labels)

                        ps = torch.exp(outputs).data
                        top_p, top_class = ps.topk(1, dim=1)

                        equality = top_class.view(*labels.shape) == labels
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

            valloss = valloss / len(validloaders)
            accuracy = accuracy /len(validloaders)

            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss {:.4f}".format(valloss),
                  "Accuracy: {:.4f}".format(accuracy*100))

            running_loss = 0


# save the model in dictionary
state = {
    'arch': arch,
    'classifier': model.classifier,
    'class_to_idx': image_datasets.class_to_idx,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'criterion': criterion.state_dict()
}

# save dictionary in json file
torch.save(state, filepath)

print("model is saved")