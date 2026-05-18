"""
WasteNet AI - Run 4: ResNet18 Optimization Training Script
Author: Yahav
Date: May 2026

Final Metrics (Run 4):
- Train Loss: 0.4742 | Train Acc: 99.85%
- Val Loss:   0.6696 | Val Acc:   91.93%
- Test Acc:   91.49%

Description: Architectural adjustment to a lighter ResNet18 model to prevent overfitting on the dataset, combined with advanced training utilities and post-analysis reporting.
"""
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch.amp as amp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = "Run4_ResNet18"
DATA_DIR = "./split_dataset"
BACKUP_DIR = "./checkpoints/Run4"
GRAPHS_DIR = os.path.join(BACKUP_DIR, 'Final_Graphs')
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

writer = SummaryWriter(f'runs/{run_name}')

hparams = {
    'img_size': 224, 'batch_size': 64, 'epochs': 150,
    'optimizer': 'AdamW', 'learning_rate': 0.001, 'weight_decay': 0.01,
    'augmentations': 'Advanced + AMP + Label Smoothing', 'patience': 15, 'model': 'ResNet18'
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([transforms.RandomResizedCrop(hparams['img_size'], scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), normalize])
test_val_transforms = transforms.Compose([transforms.Resize((hparams['img_size'], hparams['img_size'])), transforms.ToTensor(), normalize])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=test_val_transforms)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  transform=test_val_transforms)

train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=hparams['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=hparams['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
class_names = test_dataset.classes

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])
scheduler = CosineAnnealingLR(optimizer, T_max=hparams['epochs'])
scaler = torch.amp.GradScaler('cuda')

best_val_acc = 0.0
start_epoch = 0
epochs_no_improve = 0
checkpoint_path = os.path.join(BACKUP_DIR, f'{run_name}_checkpoint.pth')

for epoch in range(start_epoch, hparams['epochs']):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Run 4 - Epoch {epoch+1}/{hparams['epochs']}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100. * correct / total

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
    val_acc  = 100. * val_correct / val_total
    scheduler.step()

    writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
    writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        save_path = os.path.join(BACKUP_DIR, f'{run_name}_best.pth')
        torch.save(model.state_dict(), save_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= hparams['patience']:
            break

model.load_state_dict(torch.load(os.path.join(BACKUP_DIR, f'{run_name}_best.pth')))
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
test_acc = 100. * np.sum(all_preds == all_labels) / len(all_labels)
writer.close()
