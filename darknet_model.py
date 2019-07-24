# -*- encoding: utf8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,LeakyReLU,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.optimizers import SGD,Adam

model=Sequential()

model.add(Conv2D(8,(3,3),strides=(1,1),input_shape=(64,128,3),padding='same',kernel_initializer='random_uniform'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(LeakyReLU(alpha=0.057))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='random_uniform'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(LeakyReLU(alpha=0.151))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='random_uniform'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(LeakyReLU(alpha=0.151))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',kernel_initializer='random_uniform'))
# model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
# model.add(LeakyReLU(alpha=0.151))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_initializer='random_uniform'))
# model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
# model.add(LeakyReLU(alpha=0.151))
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-5, momentum=0.618, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)
##   LearningRate = LearningRate * 1/(1 + decay * epoch)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.save('darknet.h5')
model.summary()
