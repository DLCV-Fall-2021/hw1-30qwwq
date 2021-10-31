import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import os
from torchvision.datasets import DatasetFolder
from torchvision import models
import random
from torch.autograd import Variable
import imageio
import torchvision.transforms.functional as TF

def map_ioutarget(img):
    masks = np.empty((1, 512, 512))
    # print(files.shape[0])
    # for i, file in enumerate(files):
    # print(file.shape)
    i = 0
    # mask = imageio.imread(path)
    mask = (img >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[i, mask == 2] = 3  # (Green: 010) Forest land 
    masks[i, mask == 1] = 4  # (Blue: 001) Water 
    masks[i, mask == 7] = 5  # (White: 111) Barren land 
    masks[i, mask == 0] = 6  # (Black: 000) Unknown 
    masks[i, mask == 4] = 6  # (Black: 000) Unknown 
    # print(path, masks[i, mask == 4])

    return masks
    

class seg7_test(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        # read filenames
        self.inputs = glob.glob(os.path.join(root,'*.jpg'))
        self.inputs.sort()
        # self.targets = glob.glob(os.path.join(root, '*_mask.png'))
        # self.targets.sort() 
        self.filenames = list(self.inputs)
                
        self.len = len(self.inputs)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        # print(image_fn, label)
        image = Image.open(image_fn)
        
        image = np.array(image, dtype=np.uint8)
        image = image.astype(np.float64)
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image).float()
        image[0].add_(-0.485).div_(0.229)
        image[1].add_(-0.456).div_(0.224)
        image[2].add_(-0.406).div_(0.225)
        
        return image, image_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
