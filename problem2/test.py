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
from dataset import seg7_test
from model import FCN8s
import torch.nn.functional as F
import argparse

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mask_to_img(mask, path):
    # for mask in masks:
    img = np.zeros((3, 512, 512), dtype=np.uint8)
    img[1][mask == 0] = 255  # (Cyan: 011) Urban land 
    img[2][mask == 0] = 255  # (Cyan: 011) Urban land 
    img[0][mask == 1] = 255  # (Yellow: 110) Agriculture land
    img[1][mask == 1] = 255  # (Yellow: 110) Agriculture land 
    img[0][mask == 2] = 255  # (Purple: 101) Rangeland 
    img[2][mask == 2] = 255  # (Purple: 101) Rangeland 
    img[1][mask == 3] = 255  # (Green: 010) Forest land 
    img[2][mask == 4] = 255  # (Blue: 001) Water 
    img[0][mask == 5] = 255  # (White: 111) Barren land 
    img[1][mask == 5] = 255  # (White: 111) Barren land 
    img[2][mask == 5] = 255  # (White: 111) Barren land 
    # print(img.shape)
    imageio.imwrite(path, np.transpose(img, (1,2,0)))

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    same_seeds(30)
    batch_size = 16
    test_set = seg7_test(root=config.img_dir, transform='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,  pin_memory=True)
    
    model = models.vgg16(pretrained=True)
    n_class = 7
    fcn_model = FCN8s(pretrained_net=model, n_class=n_class)
    if torch.cuda.is_available():
        fcn_model = fcn_model.cuda()
    checkpoint = torch.load(config.model_path)
    fcn_model = checkpoint['model']
    fcn_model.eval()
    
    for batch in test_loader:
        imgs, img_path = batch

        imgs_tensor = Variable(imgs.to(device))     
        with torch.no_grad():
          logits = fcn_model(imgs_tensor)

        pred = F.log_softmax(logits, dim=1).argmax(dim=1).data.cpu().numpy()

        img_paths = [path.split('/') for path in img_path]
        for i, path in enumerate(img_paths):
            # print(path[-1][:4])
            save_path = path[-1][:-4] + ".png"
            # print(pred_list[i].min(), pred_list[i].max())
            mask_to_img(pred[i], os.path.join(config.save_dir, save_path))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='./hw1_data/p2_data/validation')
    parser.add_argument('--save_dir', type=str, default='./pred')
    parser.add_argument('--model_path', default='hw1_2_model.pth', type=str, help='model path')
    
    config = parser.parse_args()
    # print(config)
    main(config)
