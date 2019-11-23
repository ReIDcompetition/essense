#coding=utf-8
 
import torch
import os, glob
import random
 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
 
class MyDataset(Dataset):
 
    def __init__(self, root, datatxt, transform=None, target_transform=None):
 
        super(MyDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
 
        fh = open(datatxt, 'r') 
        # image, label
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.images = imgs
        self.labels = self.load_txt(datatxt)

 
    def load_txt(self, filename):
        """
        :param filename:
        :return:
        """
        labels = pd.read_csv(filename, encoding='utf-8', engine='python',header=None)
        labels = np.array(labels).reshape(1,-1)[0].tolist()
 
        return labels
 
 
    def __getitem__(self, index):
        root = self.root
        fn, label = self.images[index]
        img = Image.open(root+fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.images)

