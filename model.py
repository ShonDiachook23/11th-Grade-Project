import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 5



# Define a simple CNN model
class SimpleCNN(nn.Module):
    def _init_(self):
        super(SimpleCNN, self)._init_()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # Fully connected layer
        self.fc2 = nn.Linear(128, 3) # Output layer with 10 classes (for classification)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Conv -> ReLU -> Max Pooling
        x = self.pool(F.relu(self.conv2(x))) # Conv -> ReLU -> Max Pooling
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor
        x = F.relu(self.fc1(x)) # Fully connected -> ReLU
        x = self.fc2(x) # Fully connected -> Output
        return x

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)






# Model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad() # Zero the parameter gradients
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

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