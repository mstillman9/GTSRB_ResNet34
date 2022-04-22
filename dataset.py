import os
import json
import time
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import resnet18
import torchvision.transforms as transforms

import csv

class GTSRB(data.Dataset):
    def __init__(self, path):
        self.img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])        
        self.imgs = []
        self.lbls = []
        # Adapted from http://benchmark.ini.rub.de/Dataset/GTSRB_Python_code.zip
        for c in range(0,43):
            prefix = path + os.sep + format(c, '05d') + os.sep # subdirectory for class
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                self.imgs.append(prefix + row[0]) # the 1th column is the filename
                self.lbls.append(int(row[7])) # the 8th column is the label
            gtFile.close()        

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img = self.img_transforms(img)
        lbl = torch.tensor(self.lbls[index])
        return img, lbl

    def __len__(self):
        return len(self.imgs)    
