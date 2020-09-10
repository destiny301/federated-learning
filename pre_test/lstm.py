import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from sklearn.utils import shuffle
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''

# dataset 
total_words = 10000
max_review_len = 80
embedding_len = 100


(xtr, ytr), (xte, yte) = tf.keras.datasets.imdb.load_data(num_words=total_words)
xtr = tf.keras.preprocessing.sequence.pad_sequences(xtr, maxlen=max_review_len)
xte = tf.keras.preprocessing.sequence.pad_sequences(xte, maxlen=max_review_len)

# model
rnn_in = Input(shape=(80,))
z= tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)(rnn_in)
z = LSTM(256, return_sequences=True, stateful=False, unroll=True)(z)
z = LSTM(256, stateful=False, unroll=True)(z)
training_pred = Dense(1, activation = 'sigmoid')(z)
model = Model(inputs=rnn_in, outputs=training_pred)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

# federal
C = 0.5
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
data = np.arange(25000*80).reshape(100, 250, 80)
label = np.arange(25000).reshape(100, 250)
for k in range(100):
    data[k] = xtr[k*250:(k+1)*250]
    label[k] = ytr[k*250:(k+1)*250]


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