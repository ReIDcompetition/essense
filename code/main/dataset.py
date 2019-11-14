#coding=utf-8

import torch
import os, glob
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):

    def __init__(self, root, datatxt, mode, transform=None, target_transform=None):

        super(MyDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # self.name2label = {}

        # for name in sorted(os.listdir(os.path.join(root))):

        #     if not os.path.isdir(os.path.join(root, name)):
        #         continue

        #     self.name2label[name] = len(self.name2label.keys())

        # eg: {'squirtle': 4, 'bulbasaur': 0, 'pikachu': 3, 'mewtwo': 2, 'charmander': 1}
        # print(self.name2label)
        fh = open(root + datatxt, 'r')
        # image, label
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.images = imgs
        self.labels = self.load_txt('./data/train/train_list.txt')
        x_train, x_val, y_train, y_val = train_test_split(self.images, self.labels, test_size=0.2) #划分数据集


        if mode == "train": # 80%
            self.images = x_train
            self.labels = y_train
        else: # 20%
            self.images = x_val
            self.labels = y_val

    def load_txt(self, filename):
        """
        :param filename:
        :return:
        """
        images = pd.read_csv(filename, sep=' ', encoding='utf-8', engine='python',usecols=[0],header=None)
        images = np.array(images).reshape(1,-1)[0].tolist()
        labels = pd.read_csv(filename, encoding='utf-8', engine='python',header=None)
        labels = np.array(labels).reshape(1,-1)[0].tolist()

        return labels


    def __getitem__(self, index):
        root = './data/train/'
        fn, label = self.images[index]
        img = Image.open(root+fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.images)
