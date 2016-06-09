# basic
from keras.models import Sequential
from keras.layers import Merge, Dense

# model plot
from keras.utils.visualize_util import plot

# generate dummy data
import numpy as np
from keras.utils.np_utils import to_categorical

###########################################################
# now the model will take as input arrays of shape (*, 784)
# and output arrays of shape (*, 32)
# 左隐藏层的输入大小为784，输出大小为32 
# 28*28 = 784 ， MNIST数据的大小一般为28*28，故而多用784
left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

# for a multi-input model with 10 classes:
# 合并后的模型，输出为10，这里可以理解为一个长度为10的一维矩阵
# 或者理解为一个有10个神经元的隐藏层
# 激活函数是softmax（逻辑回归）
model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))

###########################################################
# compile
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

###########################################################
# plot the model sturcture
plot(model, show_shapes=True, to_file='model.png')

###########################################################
# construct dataset
# 模拟MNIST数据库，建立的随机数组
data_1 = np.random.random((5000, 784))
data_2 = np.random.random((5000, 784))

# these are integers between 0 and 9
labels = np.random.randint(10, size=(5000, 1))
# we convert the labels to a binary matrix of size (1000, 10)
# for use with categorical_crossentropy
# 交叉熵代价函数
labels = to_categorical(labels, 10)

###########################################################
print('start　model fit')
# train the model
# note that we are passing a list of Numpy arrays as training data
# since the model has 2 inputs
# 一次的训练样本为32，循环10个世代
# verbose 打印参数设置为传统打印 2
model.fit([data_1, data_2], labels, nb_epoch=10, batch_size=32, verbose=2)
print('end model fit')
