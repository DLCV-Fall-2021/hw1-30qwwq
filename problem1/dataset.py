import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import os
from torchvision.datasets import DatasetFolder



class classify50(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.transform = transform

        # read filenames
        for i in os.listdir(root):
            filename = os.path.join(root, i)
            label = int(i.split('_')[0])
            self.filenames.append((filename, label)) 
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
        
class classify50_test(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.transform = transform

        # read filenames
        for i in os.listdir(root):
            filename = os.path.join(root, i)
            self.filenames.append(filename) 
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, image_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
