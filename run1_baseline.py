"""
WasteNet AI - Run 1: Baseline Training Script
Author: Yahav
Date: May 2026

Final Metrics (Run 1):
- Train Loss: 0.0740 | Train Acc: 97.63%
- Val Loss:   1.4282 | Val Acc:   75.78%
- Test Acc:   79.00%

Description: Initial baseline model using ResNet50 trained from scratch (no pretrained weights) with SGD optimizer.
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = "Run1_Baseline"
DATA_DIR = "./split_dataset"
BACKUP_DIR = "./checkpoints/Run1"
os.makedirs(BACKUP_DIR, exist_ok=True)
writer = SummaryWriter(f'runs/{run_name}')

hparams = {
    'img_size': 224, 'batch_size': 32, 'epochs': 40,
    'optimizer': 'SGD', 'learning_rate': 0.01, 'weight_decay': 0.0, 'augmentations': 'None'
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize((hparams['img_size'], hparams['img_size'])), transforms.ToTensor(), normalize])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=4)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=hparams['learning_rate'])
best_val_acc = 0.0

for epoch in range(hparams['epochs']):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Run 1 - Epoch {epoch+1}/{hparams['epochs']}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
    writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(BACKUP_DIR, f'{run_name}_best.pth')
        torch.save(model.state_dict(), save_path)

model.load_state_dict(torch.load(os.path.join(BACKUP_DIR, f'{run_name}_best.pth')))
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
test_acc = 100. * test_correct / test_total
writer.add_hparams(hparam_dict=hparams, metric_dict={'hparam/val_accuracy': best_val_acc, 'hparam/test_accuracy': test_acc}, run_name="hparams")
writer.close()
