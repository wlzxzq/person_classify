# -*- encoding: utf8 -*-

import Resnet
from keras.optimizers import Adam

def buildModel():
    model = Resnet.ResnetBuilder.build_resnet_18((3,64,128),5)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)
    ##   LearningRate = LearningRate * 1/(1 + decay * epoch)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.save('Resnet18.h5')
    model.summary()
    return model

buildModel()