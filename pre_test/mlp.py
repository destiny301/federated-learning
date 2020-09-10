import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import regularizers
import numpy as np
import statistics
from tensorflow.keras import initializers
from sklearn.utils import shuffle

reg_val = 1e-4

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''

# dataset 
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# train_images.shape is (60000, 28, 28)
#test_images.shape (10000, 28, 28)
num_pixels = 28 * 28
train_images = train_images.reshape( (60000, num_pixels) ).astype(np.float32) / 255.0
test_images = test_images.reshape( (10000, num_pixels) ).astype(np.float32)  / 255.0

# model
nnet_inputs = Input(shape=(num_pixels,), name='images')
z = Dense(200, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros(), kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name='hidden1')(nnet_inputs)
z = Dense(200, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros(), kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name='hidden2')(z)
z = Dense(10, activation='softmax', kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros(), kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name='output')(z)

model = Model(inputs=nnet_inputs, outputs=z)

model.summary()
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# federal
C = 0.2
E = 10
B = 32
w = model.get_weights()

K = 100
round = 1000

def ClientUpdate(data, label, w):
    model.set_weights(w)
    model.fit(data, label, batch_size=B, epochs=E, validation_split=0.2)

    # # using a .hdf5 or .h5 extension saves the model in format compatible with older keras
    # our_model.save('fmnist_trained.hdf5')

    return model.get_weights()

#split data into k arrays
data = np.arange(60000*28*28).reshape(100, 600, 784)
label = np.arange(60000).reshape(100, 600)
for k in range(100):
    data[k] = train_images[k*600:(k+1)*600]
    label[k] = train_labels[k*600:(k+1)*600]


for i in range(round):
    m = max(C*K, 1)
    m = int(m)
    data,label = shuffle(data, label)
    S_data = data[:m]
    S_label = label[:m]
    weight = []

    for j in range(m):
        weight.append(ClientUpdate(S_data[j], S_label[j], w))
    w = np.mean(weight, axis=0)




