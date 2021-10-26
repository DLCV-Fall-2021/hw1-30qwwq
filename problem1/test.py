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

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    same_seeds(30)
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 50)
    if torch.cuda.is_available():
      model = model.cuda()
    checkpoint = torch.load(config.model_path)
    model = checkpoint['model']
    
    batch_size = 16
    test_set = classify50_test(root=config.img_dir, transform=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,  pin_memory=True)
    
    model.eval()
    predictions = []
    accs = []
    img_name = []

    for batch in test_loader:
        imgs, img_paths = batch
    
        with torch.no_grad():
            logits = model(imgs.to(device))
        
        # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        # accs.append(acc)

        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        for img_path in img_paths:
            img_name.append(img_path.split('/')) 
            
    # print(f'acc: {sum(accs) / len(accs)}')
    with open(config.save_dir, "w") as f:
        f.write("image_id,label\n")
    
        for i, pred in  enumerate(predictions):
             f.write(f"{img_name[i][-1]},{pred}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='./hw1_data/p1_data/val_50')
    parser.add_argument('--save_dir', type=str, default='./pred')
    parser.add_argument('--model_path', default='hw1_1_model.pth', type=str, help='model path.')
    
    config = parser.parse_args()
    # print(config)
    main(config)