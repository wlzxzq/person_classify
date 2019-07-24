# -*- encoding: utf8 -*-
import os
import random

dataset_dir = '/home/aaa/Desktop/projects/00000_dataset/00002_person_classify/images_person_labeled'
datalist = [os.path.join(dataset_dir,item)  for item in os.listdir(dataset_dir)]
ratio = 0.8


random.shuffle(datalist)
for data in datalist[:int(len(datalist)*ratio)]:
    cmd = "cp {} ./data/train/{}".format(data,data.split('/')[-1])
    print cmd
    os.system(cmd)
for data in datalist[int(len(datalist)*ratio):]:
    cmd = "cp {} ./data/test/{}".format(data,data.split('/')[-1])
    print cmd
    os.system(cmd)

pass