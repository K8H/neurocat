import numpy as np
import tensorflow as tf

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = len(np.unique(y_train))
input_size = list(x_train.shape[1:])

y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

# image scaling
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# less data
x_train = x_train[:1000]
y_train = y_train[:1000]

x_test = x_test[:1000]
y_test = y_test[:1000]

y_test_onehot = y_test_onehot[:1000]
y_train_onehot = y_train_onehot[:1000]

