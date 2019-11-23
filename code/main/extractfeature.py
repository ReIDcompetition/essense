# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense, PCB, PCB_test
from test_dataset import MyDataset
import json
import re_ranking

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Testing')
# parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test/',type=str, help='./test_data')
parser.add_argument('--gallery_dir',default='./data/test/gallery_a/',type=str, help='./test_gallery')
parser.add_argument('--name', default='ft_ResNet50_pcb_duke_e', type=str, help='save model path')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--use_resnet', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )

opt = parser.parse_args()

# str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ColorJitter(brightness=1.5, contrast=1.7),  # 锐化
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_datasets = MyDataset(root=opt.test_dir, datatxt='./data/test/query_a_list.txt', transform=data_transforms)
gallery_datasets = MyDataset(root=opt.gallery_dir, datatxt='./data/test/gallery_a_list.txt', transform=data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize, shuffle=False, num_workers=8)
gal_dataloaders =torch.utils.data.DataLoader(gallery_datasets, batch_size=opt.batchsize, shuffle=False, num_workers=8)
use_gpu = torch.cuda.is_available()


# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    print(save_path)
    network.load_state_dict(torch.load(save_path))
    return network


# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        if opt.use_resnet:
            ff = torch.FloatTensor(n,2048).zero_()
        else:
            ff = torch.FloatTensor(n,1024).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have four parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            if use_gpu:
                input_img = Variable(img.cuda())
            else:
                input_img = Variable(img)
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        if opt.PCB:
            # feature size (n,2048,4)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

class_num=4768

if opt.PCB:
    model_structure = PCB(class_num)

model = load_network(model_structure)
# Remove the final fc layer and classifier layer
if not opt.PCB:
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
else:
    model = PCB_test(model)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

gallery_feature = extract_feature(model,gal_dataloaders)
query_feature = extract_feature(model,dataloaders)
result = {'query_a':query_feature.numpy(),'gallery_a':gallery_feature.numpy()}
scipy.io.savemat('model/'+name+'/'+'pytorch_result.mat',result)
'''
query_list = list()
with open(r'./data/test/query_a_list.txt', 'r') as f:
    # 测试集中txt文件
    lines = f.readlines()
    for i, line in enumerate(lines):
        data = line.split(" ")
        image_name = data[0].split("/")[1]
        img_file = os.path.join(r'query_a', image_name)  # 测试集query文件夹
        query_list.append(img_file)

gallery_list = [os.path.join(r'./data/test/gallery_a', x) for x in  # 测试集gallery文件夹
                os.listdir(r'./data/test/gallery_a')]
query_num = len(query_list)
distmat = re_ranking(query_feature, gallery_feature) # rerank方法
distmat = distmat # 如果使用 euclidean_dist，不使用rerank改为：distamt = distamt.numpy()
num_q, num_g = distmat.shape
indices = np.argsort(distmat, axis=1)
max_200_indices = indices[:, :200]

res_dict = dict()
for q_idx in range(num_q):
    print(query_list[q_idx])
    filename = query_list[q_idx][query_list[q_idx].rindex("\\") + 1:]
    max_200_files = [gallery_list[i][gallery_list[i].rindex("\\") + 1:] for i in max_200_indices[q_idx]]
    res_dict[filename] = max_200_files
'''
#with open(r'submission_A.json', 'w', encoding='utf-8') as f:  # 提交文件
 #   json.dump(res_dict, f)