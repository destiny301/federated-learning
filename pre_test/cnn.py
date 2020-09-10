import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D,Flatten
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.utils import shuffle

# dataset 
((train_images, train_labels), (test_images, test_labels)) = fashion_mnist.load_data()
	 
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    
# scale data to the range of [0, 1]
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# one-hot encode the training and testing labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

#model
data = Input(shape=(28,28,1,), name='images')
z = Conv2D(32, (5, 5), activation='relu')(data)
z = MaxPooling2D((2, 2))(z)
z = Conv2D(64, (5, 5), activation='relu')(z)
z = MaxPooling2D((2, 2))(z)
z = Flatten()(z)
z = Dense(512, activation='relu')(z)
z = Dense(10, activation='softmax')(z)

model = Model(inputs=data, outputs=z)

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss=CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

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
data = np.arange(60000*28*28).reshape(100, 600, 28, 28, 1)
label = np.arange(60000*10).reshape(100, 600, 10)
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
