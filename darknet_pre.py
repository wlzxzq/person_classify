# -*- encoding: utf8 -*-
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import keras
import random
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import argparse
from keras import backend as K
K.clear_session()
from keras.applications.resnet50 import ResNet50

iw=64
ih=128
model_path='./Resnet18.h5'
train_path='data/train'
valid_path='data/test'
log_dir='logs/'
batch_size=128

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def get_image_data(image):
    im = Image.open(image)
    img = im.resize((int(iw), int(ih)), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)  # image类 转 numpy
    img = img.reshape([1, -1])
    img_data = np.array([i/255 for i in img])

    img_class = image.split('/')[-1].split('_')[-1].split('.')[0]
    dict = {"00":0,"01":1,"10":2,"11":3,"none":4}
    class_data = keras.utils.to_categorical(dict[img_class], num_classes=5)
    return img_data,class_data

def generate_arrays_from_file(path,batch_size):
    file_list = os.listdir(path)
    n=len(file_list)

    while True:
        np.random.shuffle(file_list)
        image_data = []
        class_data = []
        for b in range(batch_size):
            image_, class_ = get_image_data(os.path.join(path,file_list[random.randint(0,n-1)]))
            image_data.append(image_)
            class_data.append(class_)

        image_data = np.array(image_data).reshape([-1,iw,ih,3])
        class_data = np.array(class_data).reshape([-1,5])
        yield (image_data,class_data)

def train(model):
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:0>5d}-{loss:.3f}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.h5',
                                 monitor='val_acc', save_weights_only=False, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=30, verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=300, verbose=1)


    model.fit_generator(generator=generate_arrays_from_file(train_path,batch_size),
                        steps_per_epoch=len(os.listdir(train_path))//batch_size,
                        validation_data=generate_arrays_from_file(valid_path,batch_size),
                        validation_steps=len(os.listdir(valid_path))//batch_size,
                        epochs=int(input('Input epochs:')),
                        initial_epoch=int(input('Input initial_epochs:')),
                        verbose=1,
                        max_queue_size=2048,
                        callbacks=[logging, checkpoint, early_stopping]
                        )

def valid(model):
    score = model.evaluate_generator(generator=generate_arrays_from_file(valid_path,batch_size),
                                     steps=len(os.listdir(valid_path))//batch_size,
                                     max_queue_size=1024,
                                     workers=16,
                                     use_multiprocessing=True,
                                     verbose=1)
    print score


def predict(model):
    while True:
        img_path = input("input path of img: ")
        im = Image.open(img_path)
        img = im.resize((int(iw), int(ih)), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)  # image类 转 numpy
        img = img.reshape([1, -1])
        img_data = np.array([i / 255 for i in img]).reshape([-1,iw,ih,3])
        predict = model.predict(img_data)
        print predict



def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-t','--train',action='store_true',help='train')
    argparser.add_argument('-v','--valid', action='store_true', help='valid')
    argparser.add_argument('-p', '--predict', action='store_true', help='predict')
    args = argparser.parse_args()

    model = load_model(model_path)

    if args.train:
        train(model)
    if args.valid:
        valid(model)
    if args.predict:
        predict(model)



if __name__ == '__main__':
    main()