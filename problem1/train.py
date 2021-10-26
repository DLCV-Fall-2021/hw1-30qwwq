import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
from torch.utils.data import Dataset, DataLoader
import glob
import os
from torchvision.datasets import DatasetFolder
from torchvision import models
from dataset import classify50_test
from dataset import classify50

train_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224,224)),
    # transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    same_seeds(30)
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 50)
    if torch.cuda.is_available():
      model = model.cuda()
    
    batch_size = 16
    train_set = classify50(root='./hw1_data/p1_data/train_50', transform=train_tfm)
    valid_set = classify50(root='./hw1_data/p1_data/val_50', transform=test_tfm)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    t = len(train_loader)*5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t, eta_min=0)
    n_epochs = 5
    best_acc = 0
    
    for epoch in range(n_epochs):
        model.train()
    
        train_loss = []
        train_accs = []
    
        for batch in train_loader:
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            # print(logits, labels)
            optimizer.zero_grad()
            loss.backward()
    
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
    
            optimizer.step()
            scheduler.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # print(acc)
            train_loss.append(loss.item())
            train_accs.append(acc)
    
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
    
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    
        # ---------- Validation ----------
        model.eval()
    
        valid_loss = []
        valid_accs = []
    
        for batch in valid_loader:
            imgs, labels = batch
    
            with torch.no_grad():
              logits = model(imgs.to(device))
    
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    
            valid_loss.append(loss.item())
            valid_accs.append(acc)
    
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
    
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if (valid_acc>best_acc):
          best_acc = valid_acc
          checkpoint = {'model': model,
                  'optimizer': optimizer,
                  'lr_sched': scheduler}
          torch.save(checkpoint, 'model_hw1_1.pth')
          print('save checkpoint!\n')


if __name__ == '__main__':
    main()