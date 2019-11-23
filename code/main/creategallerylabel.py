import cv2
import os

dirl = input('请输入需要转化的文件夹名：')

daddir = './'
path = daddir + dirl
path_list = os.listdir(path)
number = 0  # 统计图片数量
for filename in path_list:
    number += 1
    f = open('./data/test/gallery_a_list.txt', 'a')
    f.write(str(filename) + ' '+ str(number) +'\n' )
    f.close()



