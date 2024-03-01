import sys, os

import numpy as np

sys.path.append(os.pardir)
from common.function import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2, b1, b2 = self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1)+b1
        y1 = sigmoid(a1)
        a2 = np.dot(y1,W2)+b2
        y2 = softmax(a2)

        return y2

    def loss(self,x,t):
        y=self.predict(x)

        return cross_entropy_error(y,t)


    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1)
        #numpy.argmax是NumPy库中的一个函数，用于返回数组中最大值的索引。如果数组有多个维度，你可以指定在哪个轴上查找最大值的索引。
        #在二维数组中，axis=0 表示沿着行的方向查找每列的最大值，而 axis=1 表示沿着列的方向查找每行的最大值。

        true=np.sum(y==t)
        accuracy=true/float(x.shape[0])

        return accuracy

    def numerical_gradient(self,x,t):
        loss_w = lambda w: self.loss(x,t)

        grads={}    #声明字典保存梯度
        grads['W1'] = numerical_gradient(loss_w,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w,self.params['b2'])

        return grads

