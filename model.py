import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from collections import defaultdict
from random import sample

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 5
image_size = (64, 64)  # Define a fixed size for the images

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjusted for 64x64 input size
        self.fc2 = nn.Linear(128, 2)  # Output layer with 2 classes (cat and dog)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv -> ReLU -> Max Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv -> ReLU -> Max Pooling
        x = x.view(-1, 64 * 16 * 16)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Fully connected -> ReLU
        x = self.fc2(x)  # Fully connected -> Output
        return x

class LimitedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, limit_per_class=1000):
        super(LimitedImageFolder, self).__init__(root, transform=transform)
        self.limit_per_class = limit_per_class
        self.samples = self._limit_samples()

    def _limit_samples(self):
        class_counts = defaultdict(list)
        for path, class_idx in self.samples:
            class_counts[class_idx].append((path, class_idx))
        
        limited_samples = []
        for class_idx, items in class_counts.items():
            if len(items) > self.limit_per_class:
                limited_samples.extend(sample(items, self.limit_per_class))
            else:
                limited_samples.extend(items)
        
        return limited_samples

# Data preparation
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure the images are in grayscale
    transforms.Resize(image_size),  # Resize the images to a fixed size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = LimitedImageFolder(root='./train', transform=transform, limit_per_class=1000)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = LimitedImageFolder(root='./test1', transform=transform, limit_per_class=1000)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Finished Training')

# Testing the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
