import scipy.io
import torch
import numpy as np
import time
from  re_ranking import re_ranking
import os
import torch
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument('--name',default='ft_ResNet50_pcb_duke_e', type=str, help='0,1,2,3...or last')
opt = parser.parse_args()
name = opt.name

result = scipy.io.loadmat('model/'+name+'/pytorch_result.mat')
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

query_feature = result['query_a']
gallery_feature = result['gallery_a']
q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
q_q_dist = np.dot(query_feature, np.transpose(query_feature))
g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist)
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

with open(r'submission_A.json', 'w', encoding='utf-8') as f:  # 提交文件
    json.dump(res_dict, f)


