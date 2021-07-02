# Fine-tune a pretrained ResNet50 model for image classification
#
#
# Usage:
# python ml_script.py --data_dir /home/user/data --model_dir /home/user/model


# Imports
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--model_dir', type=str, default='/home/user/model', help='directory to store the model')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
args = parser.parse_args()

# Hyperparameters
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
MOMENTUM = args.momentum
NUM_WORKERS = args.num_workers
BATCH_SIZE = args.batch_size
GPU = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

# Define pretrained ResNet50 model class for fine-tuning
class ResNet50(nn.Module):
    def __init__(self, num_classes=102):       
        super(ResNet50, self).__init__()       
        self.model = models.resnet50(pretrained=True)       
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Data augmentation and normalization for training dataset
# Just normalization for validation dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load training and validation datasets with ImageFolder
image_datasets = {
    x: datasets.ImageFolder(root=os.path.join(args.data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']
}

# Define dataloaders for training and validation dataset
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) for x in ['train', 'val']}

# Train and evaluate the model
model = ResNet50()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optimizer.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
since = time.time()
for epoch in range(NUM_EPOCHS):
    print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
    print('-' * 10)
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train mode
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


