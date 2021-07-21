# PyTorch training script for fine-tuning pretrained ResNet50 on a custom dataset. The dataset is provided as a folder of images 
# and a .csv file containing the corresponding labels.
# Usage: 
#   python fine_tune_resnet50.py --dataset [path to custom dataset] --model [path to pre-trained model]
# Steps:
# 1. Split dataset into training and validation sets
# 2. Define PyTorch Dataset and DataLoader (with augmentations) for both training and validation sets
# 3. Define the ResNet50 model and load the pretrained weights.
# 4. Train the classifier model using the pre-trained ResNet50 model as its backbone, and the new dataset as input, for 
# a few epochs
# 5. Evaluate the classifier model using the validation dataset, and print its accuracy
# 6. Save the model and the weights to disk

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Split dataset into training and validation sets

def split_dataset(dataset_csv, valid_pct=0.2):
    """
    Given the CSV file, split into a train set CSV and a validation set csv
    """
    # Read the dataset
    df = pd.read_csv(dataset_csv)
    # Split into training and validation sets
    n_valid = int(valid_pct * len(df))
    valid_df = df.iloc[:n_valid]
    train_df = df.iloc[n_valid:]
    # Save the new CSVs
    valid_df.to_csv('valid.csv', index=False)
    train_df.to_csv('train.csv', index=False)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2: Define PyTorch Dataset and DataLoader (with augmentations) for both training and validation sets

class CustomDataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for loading the custom dataset from CSV with the image path in the column"""

    def __init__(self, csv_path, transform=None):                   
        """
        Args:
            csv_path (string): Path to the csv file with annotations
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)                   
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])           
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])           
        # Transformations
        self.transform = transform

    def __getitem__(self, index):                                   
        """
        Args: index (int): Index
        Returns: image, target (tuple): where target is index of the target class.
        """
        # Read the image from disk
        image = Image.open(self.image_arr[index])
        # Convert to PyTorch tensor
        image = self.transform(image)
        # Return sample
        return image, self.label_arr[index]

    def __len__(self):                                                
        """
        Returns: len (int): Total number of samples in the dataset
        """
        return len(self.data_info)

def get_data_loader(dataset_csv, batch_size, augment=True):
    """
    Utility function for loading the custom dataset with PyTorch, along with a data loader
    Args: dataset_csv (string): Path to the csv file with annotations
        batch_size (int): Batch size
        augment (boolean): Flag to turn on data augmentation
    Returns: data_loader (torch.utils.data.DataLoader): Data loader for PyTorch custom dataset
    """
    # Load the dataset
    dataset = CustomDataset(dataset_csv, transform=augment)                
    # Create the data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 3: Define the ResNet50 model and load the pretrained weights

# Define the ResNet50 model
class ResNet50(nn.Module):                   
    def __init__(self, num_classes=2):                   
        """
        Initialize the model descriptor with a sequential model type
        Args: num_classes (int): Number of classes in the dataset
        """                   
        super(ResNet50, self).__init__()                   
        # Define the network architecture                   
        self.net = torchvision.models.resnet50(pretrained=True)                   
        # Freeze the network weights
        for param in self.net.parameters():
            param.requires_grad = False
        # Define the new network with the same architecture as the pretrained network, except without last fully connected layer                   
        self.new_net = nn.Sequential(*list(self.net.children())[:-1])
        # Add the new linear layer with num_classes output classes
        self.new_net.add_module('fc_new', nn.Linear(2048, num_classes))
        # Initialize the weights in the new linear layer
        nn.init.normal_(self.new_net.fc_new.weight, std=0.001)
        nn.init.constant_(self.new_net.fc_new.bias, 0)
                
    def forward(self, x):                   
        """
        Forward pass of the network
        Args: x (torch.autograd.Variable): Input data
        Returns: y_pred (torch.autograd.Variable): Output data
        """                   
        # Forward pass of the pretrained network
        x = self.net(x)
        # Forward pass of the new network
        x = self.new_net(x)
        # Return the new network
        return x

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 4: Train the classifier model using the pre-trained ResNet50 model as its backbone, and the new dataset as input, for a few epochsPytorch

# Define the ResNet50 model
model = ResNet50(num_classes=2)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# Define the training and validation datasets
train_loader = get_data_loader('train.csv', batch_size=32, augment=True)
valid_loader = get_data_loader('valid.csv', batch_size=32, augment=False)
# Train the model for a few epochs
num_epochs = 10
for epoch in range(num_epochs):
    # Start the training model
    model.train()                   
    # Loop over the training dataset
    for i, (images, labels) in enumerate(train_loader):                   
        # Clear the gradients from all Variables
        optimizer.zero_grad()                   
        # Run the forward pass
        outputs = model(images)                   
        # Compute the loss
        loss = criterion(outputs, labels)                   
        # Backward pass and update the weights
        loss.backward()                   
        optimizer.step()                   
        # Print loss
        if (i+1) % 100 == 0:                   
            print('Epoch [{}/{}], Iter [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))                   
    # Validate the model
    model.eval()                   
    # Loop over the validation dataset
    total_correct = 0
    total_samples = 0
    for images, labels in valid_loader:                   
        # Run the forward pass
        outputs = model(images)                   
        # Compute the loss
        loss = criterion(outputs, labels)                   
        # Compute the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)                   
        total_correct += (predicted == labels).sum().item()                   
        total_samples += labels.size(0)                   
    # Print loss
    print('Epoch [{}/{}], Accuracy: {:.4f}'.format(epoch+1, num_epochs, total_correct/total_samples))                   
    # Save the model
    torch.save(model.state_dict(), 'model_resnet50_epoch_{}.ckpt'.format(epoch+1))                   
    # Update the learning rate
    scheduler.step()                   
    # Print the learning rate
    print('Current learning rate: {}'.format(scheduler.get_lr()))
    # End the training model

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
