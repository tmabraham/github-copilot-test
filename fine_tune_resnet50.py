# Training script for fine-tuning pretrained ResNet50 on a new dataset.
# Steps:
# 1. Given a URL, download the dataset
# 2. Define PyTorch Dataset and DataLoader (with augmentations) for both training and validation sets
# 3. Define the ResNet50 model and load the pretrained weights.
# 4. Train the classifier model using the pre-trained ResNet50 model as its backbone, and the new dataset as input, for a few epochs
# 5. Evaluate the classifier model using the validation dataset, and print its accuracy
# 6. Save the model and the weights to disk

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Imports

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import random
import time
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the ResNet50 model and load the pretrained weights.

class ResNet50(nn.Module):       
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes       
        resnet50 = torchvision.models.resnet50(pretrained=True)       
        self.base = nn.Sequential(*list(resnet50.children())[:-2])       
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False, count_include_pad=True)       
        self.fc = nn.Linear(2048, num_classes)       
        self.fc.apply(self.weights_init)       
        self.bn = nn.BatchNorm1d(num_classes, eps=0.001, momentum=0.1, affine=True)       
        self.bn.apply(self.weights_init)
    
    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)       
            nn.init.constant_(m.bias, 0)
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define PyTorch Dataset and DataLoader for training and validation sets, and DataLoader for training and validation batches.

class ResNet50_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):       
        self.data_dir = data_dir       
        self.transform = transform       
        self.image_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.jpg')]       
        self.labels = [int(f.split('/')[-1].split('.')[0]) for f in self.image_files]       
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)
    
def get_data_loaders(data_dir, train_batch_size, valid_batch_size, train_transform, valid_transform):
    train_dataset = ResNet50_Dataset(data_dir
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)       
    valid_dataset = ResNet50_Dataset(data_dir, valid_transform)       
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)       
    return train_loader, valid_loader

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initialize the ResNet50 model and define the loss function and optimizer. 
    
resnet50 = ResNet50(num_classes=2)       
resnet50.cuda()       
resnet50.train()       
criterion = nn.CrossEntropyLoss()       
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)       
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
# Train the classifier model using the pre-trained ResNet50 model as its backbone, and the new dataset as input, for a few epochs.       

train_loader, valid_loader = get_data_loaders('../../data/', 64, 16, transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))       

for epoch in range(10):       
    running_loss = 0.0       
    for i, data in enumerate(train_loader, 0):       
        inputs, labels = data       
        inputs, labels = inputs.cuda(), labels.cuda()       
        optimizer.zero_grad()       
        outputs = resnet50(inputs)       
        loss = criterion(outputs, labels)       
        loss.backward()       
        optimizer.step()       
        running_loss += loss.item()       
        if i % 100 == 99:       
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))       
            running_loss = 0.0       
    correct = 0       
    total = 0       
    with torch.no_grad():       
        for data in valid_loader:       
            images, labels = data       
            images, labels = images.cuda(), labels.cuda()       
            outputs = resnet50(images)       
            _, predicted = torch.max(outputs.data, 1)       
            total += labels.size(0)       
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Save the model checkpoint.

torch.save(resnet50.state_dict(), 'resnet50.ckpt')
