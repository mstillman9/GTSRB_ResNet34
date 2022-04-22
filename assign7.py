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

from dataset import GTSRB
from model import net

model = net()
if torch.cuda.is_available():
    model = model.cuda()

train_path = os.path.join(os.path.dirname(__file__), 'GTSRB' + os.sep + 'Final_Training' + os.sep + 'Images')
dataset = GTSRB(path=train_path)

train_size = int(0.8 * len(dataset))
val_size = int(len(dataset) - train_size)
train_dataset, val_dataset = data.dataset.random_split(dataset, [train_size, val_size])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=64)
val_dataloader = data.DataLoader(val_dataset, shuffle=True, batch_size=1)

for epoch in range(20):
    print("Training...")
    model.train()

    for i, (imgs, lbls) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            lbls = lbls.cuda()

        output = model(imgs)
        loss = loss_fn(output, lbls)
        loss.backward()
        optimizer.step()

        print("Epoch " + str(epoch+1) + " train: " + str(i+1) +  "/" + str(len(train_dataloader)) + " loss: " + str(loss.item()) + "\r")

    print("Validating...")
    model.eval()
    correct = 0

    for imgs, lbls in val_dataloader:

        output = model(imgs)        
        if torch.equal(torch.argmax(output, dim=1), lbls):
            correct += 1        

    print("Epoch " + str(epoch+1) + " validation accuracy: " + str(100 * correct / len(val_dataloader)) + "%")
    torch.save(model.state_dict(), 'model' + str(epoch+1) + '-' + str(100 * correct / len(val_dataloader)) + '.pth')