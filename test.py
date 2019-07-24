# -*- encoding: utf8 -*-
from keras.models import load_model
from PIL import Image
import os
import numpy as np
import cv2

iw=64
ih=128
model_path='/home/aaa/Desktop/projects/00002_person-classify/logs/darknet_ep00043-0.008-0.998-0.625-0.890.h5'
train_path='data/train'
valid_path='data/train'

model = load_model(model_path)

image_list = [os.path.join(valid_path,item) for item in os.listdir(valid_path)]
image_list.sort()
dict = ["00","01","10","11","none"]
for image in image_list:
    im = Image.open(image)
    img = im.resize((int(iw), int(ih)), Image.BILINEAR)

    img = np.array(img, dtype=np.float32)  # image类 转 numpy
    img = img.reshape([1, -1])
    img_data = np.array([i / 255 for i in img]).reshape([-1, iw, ih, 3])
    predict = model.predict(img_data).reshape([-1])
    predict = list(predict)
    predict_index = predict.index(max(predict))
    label = dict[predict_index]

    if label != image.split("/")[-1].split(".")[0].split("_")[-1]:
        print image.split('/')[-1],dict[predict_index]