import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torchmetrics import Accuracy, F1Score
from PIL import Image
import os

dataset_path = 'D:/Projects/X-Ray Classification/chestxrays'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std)
])

train_dataset = ImageFolder(train_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

resnet50.fc = nn.Linear(resnet50.fc.in_features, 1)

optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

def train(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            labels = labels.float().unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            preds = torch.sigmoid(outputs) > 0.5
            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = running_accuracy.double() / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

train(resnet50, train_loader, criterion, optimizer, num_epochs=10)

def evaluate(model, test_loader):
    model.eval()
    accuracy_metric = Accuracy(task="binary")
    f1_metric = F1Score(task="binary")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).round()
            
            preds = preds.squeeze(1)
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    
    test_accuracy = accuracy_metric(all_preds, all_labels).item()
    test_f1_score = f1_metric(all_preds, all_labels).item()
    
    print(f"Test Accuracy: {test_accuracy:.3f}, Test F1 Score: {test_f1_score:.3f}")

evaluate(resnet50, test_loader)
