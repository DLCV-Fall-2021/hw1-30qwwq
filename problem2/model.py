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
import torch.nn.functional as F
import argparse

class FCN8s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)


    def forward(self, x):
        ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))
        output = {}
        for idx in range(len(ranges)):
            for layer in range(ranges[idx][0], ranges[idx][1]):
                x = self.pretrained_net.features[layer](x)
            output["x%d" % (idx+1)] = x

        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]
        x3 = output['x3']  # size=[n, 512, x.h/8, x.w/8]

        score = self.relu(self.deconv1(x5))                  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x4)                         # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))               # size=[n, 256, x.h/8, x.w/8]
        score = self.bn2(score+x3)
        score = self.bn3(self.relu(self.deconv3(score)))     # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))     # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))     # size=[n, 32, x.h, x.w]
        score = self.classifier(score)                       # size=[n, n_class, x.h, x.w]

        return score