# Training script for fine-tuning pretrained ResNet50 on a new dataset.
# Steps:
# 1. Given a URL, download the dataset
# 2. Define PyTorch Dataset and DataLoader (with augmentations) for both training and validation sets
# 3. Define a new, untrained ResNet50 model, and copy pretrained weights onto it
# 4. Define a new, untrained classifier model (a classifier that takes ResNet50 output as input)
# 5. Train the classifier model using the pre-trained ResNet50 model as its backbone, and the new dataset as input, for a few epochs
# 6. Evaluate the classifier model using the validation dataset, and print its accuracy
# 7. Save the model and the weights to disk

# -------------------------------------------------------------------------------------------- #

# Imports
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import time
import copy
import argparse
import json
import matplotlib.pyplot as plt
import PIL


# Given the URL of a dataset, download it and return the file path
def load_dataset(url, dest_data_dir):
    # Set the filename for saving the dataset locally
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_data_dir, filename)
    # Check if the file already exists
    if not os.path.exists(filepath):
        # Download the file from the given URL
        print("Downloading {}".format(url))
        data = urllib.request.urlopen(url)
        with open(filepath, 'wb') as f:
            f.write(data.read())
        print("Downloaded {}".format(filename))
    else:
        print("{} already exists".format(filename))
    return filepath




def train_func():
    # Define the device for this PyTorch program   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    # Define the hyperparameters for our network       
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    # Load the pretrained ResNet50 model from PyTorch, and put it on the GPU       
    model = models.resnet50(pretrained=True)       
    model.to(device)       
    # Define a new, untrained classifier       
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)       
    # Define a loss function and optimizer       
    criterion = nn.CrossEntropyLoss()       
    optimizer = optimizer.Adam(model.parameters(), lr=learning_rate)       
    # Load the dataset       
    data_dir = 'data'       
    train_dir = 'train'       
    valid_dir = 'valid'       
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),       
                                           transforms.RandomHorizontalFlip(),       
                                           transforms.ToTensor(),       
                                           transforms.Normalize([0.485, 0.456, 0.406],       
                                                                [0.229, 0.224, 0.225])])       
    valid_transforms = transforms.Compose([transforms.Resize(256),       
                                           transforms.CenterCrop(224),       
                                           transforms.ToTensor(),       
                                           transforms.Normalize([0.485, 0.456, 0.406],       
                                                                [0.229, 0.224, 0.225])])       
    # Load the datasets with ImageFolder   
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, train_dir), transform=train_transforms)       
    valid_dataset = datasets.ImageFolder(root=os.path.join(data_dir, valid_dir), transform=valid_transforms)
    # Define the dataloaders for the training and validation sets
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    # Loop over the dataset for num_epochs times, and train the classifier
    for epoch in range(num_epochs):
        # Run the training data through the network       
        model.train()       
        for inputs, labels in trainloader:       
            inputs = inputs.to(device)       
            labels = labels.to(device)       
            # Clear the gradients from the network       
            optimizer.zero_grad()       
            # Run the data through the network       
            outputs = model(inputs)       
            # Calculate the loss       
            loss = criterion(outputs, labels)       
            # Backpropagate the loss       
            loss.backward()       
            # Update the weights       
            optimizer.step()       
        # Run the validation data through the network       
        model.eval()       
        # Turn off gradients for validation, to save memory       
        with torch.no_grad():       
            correct = 0       
            total = 0       
            for inputs, labels in validloader:       
                inputs = inputs.to(device)       
                labels = labels.to(device)       
                # Run the data through the network       
                outputs = model(inputs)       
                # Get the predicted class from the maximum value in the output-list of class scores       
                _, predicted = torch.max(outputs.data, 1)       
                # Total number of labels in the validation set, to calculate the accuracy       
                total += labels.size(0)       
                # Total correct predictions       
                correct += (predicted == labels).sum().item()       
            print("Epoch {}/{}.. ".format(epoch + 1, num_epochs),
                  "Training Loss: {:.3f}.. ".format(loss.item()),
                  "Validation Accuracy: {:.3f}%".format(100 * correct / total))