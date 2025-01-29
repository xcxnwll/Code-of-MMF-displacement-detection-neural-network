import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Check device availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Path
train_test_data_dir = '/content/gdrive/MyDrive/imaging/classification/train'  # Modify to actual path
val_data_dir = '/content/gdrive/MyDrive/imaging/classification/val'  # Modify to actual path
save_path = '/content/gdrive/MyDrive/imaging/classification/Net.pth' # Modify to actual path
result_path = '/content/gdrive/MyDrive/imaging/classification/results/training_plot.png' # Modify to actual path

# Defining data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Modify the dimensions
    transforms.Grayscale(),         # Convert to grayscale
    transforms.ToTensor(),          # Transform into a tensor
    transforms.Normalize(mean=0.5, std=0.5)  #  Standardization
])

# Load training and test datasets (same folder)
train_test_dataset = ImageFolder(root=train_test_data_dir, transform=transform)

# Load validation dataset
val_dataset = ImageFolder(root=val_data_dir, transform=transform)

# Setting the ratio of training and test datasets
train_ratio = 0.7
train_size = int(train_ratio * len(train_test_dataset))
val_size = len(train_test_dataset) - train_size

# Scaling datasets using 'random_split'
train_dataset, val_dataset_split = random_split(train_test_dataset, [train_size, val_size])

# Creating a Data Loader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader_split = DataLoader(val_dataset_split, batch_size=256, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Defining Convolutional Neural Networks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.act3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.act4 = nn.Tanh()
        self.pool4 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)  # Add Dropout Layer to Reduce Overfitting
        self.fc1 = nn.Linear(256 * 8 * 8, 64)
        self.act5 = nn.Tanh()
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(x)
        x = self.act5(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function and optimizer
model = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialization lists, which are used to store training and validation losses and accuracies
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_losses.append(train_loss / len(train_loader))
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Evaluating models on segmented validation sets
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader_split:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(val_loader_split))
    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss / len(val_loader_split):.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Evaluate the model on an independent test set
model.eval()
test_loss = 0
correct_test = 0
total_test = 0
true_labels = []
pred_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

test_accuracy = correct_test / total_test
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.4f}')

# Saving Model
torch.save(model.state_dict(), save_path)

# Plotting loss and accuracy images for training and validation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(result_path)
print(f"Plot saved to {result_path}")

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Get category name
class_names = train_test_dataset.classes  # 从 train_test_dataset 获取类别名称

# Drawing the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.tight_layout()  # Ensure that no content is cropped
confusion_matrix_path = '/content/gdrive/MyDrive/imaging/classification/results/confusion_matrix.png' # Modify to actual path
plt.savefig(confusion_matrix_path)
plt.show()
print(f"Confusion matrix saved to {confusion_matrix_path}")
