import numpy as np
from common.layers import *
from collections import OrderedDict
from common.function import numerical_gradient
import pickle

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # weight_init_std 参数在这里的作用是指定用于初始化网络权重的标准差
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLu1'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

        # 添加一个方法来加载权重
    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            saved_weights = pickle.load(f)
            # 更新网络层的权重和偏置
            self.params['W1'] = saved_weights['W1']
            self.params['b1'] = saved_weights['b1']
            self.params['W2'] = saved_weights['W2']
            self.params['b2'] = saved_weights['b2']
            self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
            self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

    def predict(self, x):
        for value in self.layers.values():
            x = value.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:  #处理标签为one-hot情况
            t = np.argmax(t, axis=1)
        return np.sum(y == t)/y.shape[0]

    #数值微分法求梯度
    def numerical_gradient(self, x, t):
        grads = {}
        grads['W1'] = numerical_gradient(self.loss(x, t), self.params['W1'])
        grads['b1'] = numerical_gradient(self.loss(x, t), self.params['b1'])
        grads['W2'] = numerical_gradient(self.loss(x, t), self.params['W2'])
        grads['b2'] = numerical_gradient(self.loss(x, t), self.params['b2'])
        return grads


    def gradient(self, x, t):
        self.loss(x, t)  #forward
        # 确保了前向传播和反向传播的一致性。这意味着在反向传播时，我们使用的是与前向传播相同的输出和内部状态来计算梯度。

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads



