from keras.layers import Conv2D, BatchNormalization, Dense, Flatten,\
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Input, add
from keras.models import Model
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import Adam
def conv_block(inputs,
        neuron_num,
        kernel_size,
        use_bias,
        padding= 'same',
        strides= (1, 1),
        with_conv_short_cut = False):
    conv1 = Conv2D(
        neuron_num,
        kernel_size = kernel_size,
        activation= 'relu',
        strides= strides,
        use_bias= use_bias,
        padding= padding
    )(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)

    conv2 = Conv2D(
        neuron_num,
        kernel_size= kernel_size,
        activation= 'relu',
        use_bias= use_bias,
        padding= padding)(conv1)
    conv2 = BatchNormalization(axis = 1)(conv2)

    if with_conv_short_cut:
        inputs = Conv2D(
            neuron_num,
            kernel_size= kernel_size,
            strides= strides,
            use_bias= use_bias,
            padding= padding
            )(inputs)
        return add([inputs, conv2])

    else:
        return add([inputs, conv2])

inputs = Input(shape= [64, 128, 3])
x = ZeroPadding2D((3, 3))(inputs)

# Define the converlutional block 1
x = Conv2D(64, kernel_size= (7, 7), strides= (2, 2), padding= 'same')(x)
x = BatchNormalization(axis= 1)(x)
x = MaxPooling2D(pool_size= (3, 3), strides= (2, 2), padding= 'same')(x)

# Define the converlutional block 2
x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)

# Define the converlutional block 3
x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)

# Define the converlutional block 4
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)

# Define the converltional block 5
x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
x = AveragePooling2D(pool_size=(3, 5))(x)
x = Flatten()(x)
x = Dense(5, activation='softmax')(x)

model = Model(inputs= inputs, outputs= x)
# Print the detail of the model
model.summary()
# compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.save("Resnet34.h5")
