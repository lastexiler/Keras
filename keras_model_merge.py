# basic
from keras.models import Sequential
from keras.layers import Merge, Dense

# model plot
from keras.utils.visualize_util import plot

# generate dummy data
import numpy as np
from keras.utils.np_utils import to_categorical

####################################################################
# for a multi-input model with 10 classes:
left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))

####################################################################
# compile
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

####################################################################
# plot the model sturcture
plot(model, show_shapes=True, to_file='model.png')

####################################################################
# construct dataset
data_1 = np.random.random((5000, 784))
data_2 = np.random.random((5000, 784))

# these are integers between 0 and 9
labels = np.random.randint(10, size=(5000, 1))
# we convert the labels to a binary matrix of size (1000, 10)
# for use with categorical_crossentropy
labels = to_categorical(labels, 10)

####################################################################
print('start　model fit')
# train the model
# note that we are passing a list of Numpy arrays as training data
# since the model has 2 inputs
model.fit([data_1, data_2], labels, nb_epoch=10, batch_size=32, verbose=2)
print('end model fit')