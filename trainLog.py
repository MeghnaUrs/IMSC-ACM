# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:27:06 2019

@author: 12134
"""

import torchvision
import torch
from torchvision import transforms
from torch import nn
import torch.optim as optim
from autoencoder import Autoencoder
from torch.autograd import Variable
import os
import numpy as np
import glob
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import logging


zsize = 48
batch_size = 16
iterations =  20
learningRate= 0.0001

class Dataset(data.Dataset):
    def __init__(self, datapath):
        types = ('*.jpg','*.jpeg','*.JPG')
        self.all_data=[]
        for files in types:
            self.all_data.extend(glob.glob(datapath + files))


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(self.all_data[index])
        img = img.convert('RGB')
        img = trans(img)
        img = img.type(torch.FloatTensor)

        return (img,img)

data_path = '/home/ubuntu/acm_mm_competition2019/dataset/images/smalldata/'
training_dataset = Dataset(data_path)
trainloader = data.DataLoader(training_dataset, batch_size = batch_size, num_workers = 8)

autoencoder_criterion = nn.MSELoss()

autoencoder = Autoencoder()
logging.basicConfig(filename='training_1.log', filemode='w',format='%(asctime)s - %(message)s',datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO)

autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr = learningRate)
list_a_loss = []

logging.info("Starting training for {} epoch(s)".format(iterations))

for epoch in range(iterations):
    epoch_losses = []
    run_loss=0
    max_steps = 2
    train_iterator = tqdm(trainloader, total=max_steps // batch_size + 1)
    autoencoder.train(True) # For training

    for i,j in train_iterator:

        inputs, labels = i,j
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        autoencoder_optimizer.zero_grad()

        pred = autoencoder(inputs)

        a_loss = autoencoder_criterion(pred , inputs)
        epoch_losses.append(a_loss.item())
        status = 'Epoch[{0}] loss = {1:0.5f} avg = {2:0.5f}'.format(epoch + 1, a_loss.item(), np.mean(epoch_losses))

        train_iterator.set_description(status)
        logging.info(status)
        a_loss.backward()
        autoencoder_optimizer.step()
        
    if (epoch+1)%50 == 0:
        autoencoder_path = os.path.join('/home/ubuntu/meghana/IMSC-ACM/AutoencoderComplete/', 'autoenc-%d' %(epoch+1))
        torch.save(autoencoder.state_dict(), autoencoder_path)

print('Finished Training and Testing')
    