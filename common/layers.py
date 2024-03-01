import numpy as np
from common.function import *


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        return x+y

    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy

class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        x[self.mask] = 0
        return x

    def backward(self,dout):
        dout[self.mask] = 0
        return dout

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1/1+np.exp(-x)
        self.out = out
        return out

    def backward(self, dout):
        return dout*self.out*(1-self.out)



class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None

        self.original_x_shape = None

        self.dW = None   #记录权重的偏导
        self.db = None

    def forward(self, x):
        # x为张量情况下的处理
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

#卷积层
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        #中间数据 backward时使用
        self.x = None
        self.col = None
        self.col_W = None

        #权重和偏置的参数
        self.dW = None
        self.db = None

    def forward(self, x):
        N, C, H, W = x.shape
        FN, C, FH, FW = self.W.shape
        out_h = int((H+2*self.pad-FH)/self.stride + 1)
        out_w = int((W+2*self.pad-FW)/self.stride + 1)

        col = im2col(x, FH, FW, self.stride, self.pad)  #N*OH*OW , C*FH*FW
        col_W = self.W.reshape(FN, -1).T        #C*FH*FW , FN

        out = np.dot(col, col_W) + self.b  #N*OH*OW,FN
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out              #N, FN, OH, OW

    def backward(self, dout):  #dout的形状与前向传播out相同  N,FN,OH,OW
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)  #N*OH*OW, FN

        self.db = np.sum(dout, axis=0) #偏置个数FN
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.pad, self.stride)

        #把dcol还原为img特征图展开前的形式

        return dx




#池化层
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None  #存储前向传播时的输入
        self.arg_max = None     #存储最大值的位置

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int((H - self.pool_h) / self.stride + 1)
        out_w = int((W - self.pool_w) / self.stride + 1)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)  #return (N*OH*OW, C*PH*PW)
        col = col.reshape(N*out_h*out_w*C, -1)
        #col  (N*out_h*out_w*C, PH*PW)   池化层的处理在通道方向上是独立的

        # 找到每个池化窗口的最大值及其索引
        arg_max = np.argmax(col, axis=1) #记录最大值的索引
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out  #N,C,OH,OW


    '''
    对于最大池化层，反向传播的实现相对简单。在前向传播过程中，最大池化层会记录每个输出特征图上的最大值的位置（即最大索引）。
    在反向传播过程中，当接收到来自上一层的梯度时，只需将梯度值放在对应最大索引的位置，其他位置置为零。这样，梯度只会传递到
    前一层特征图上对应最大值的位置。为了实现最大池化的反向传播，可以采用与卷积层相似的技巧，即使用im2col操作将输入特征图转
    化成一个矩阵，每个窗口内的数据都对应到矩阵的每一行。这样，窗口内的最大值操作就变成矩阵的行操作，从而能够利用numpy等库进
    行高效的操作。在进行反向传播时，再对变换后的矩阵在最大值反向传播之后，使用col2im操作还原为与前向传播时输入特征图相同的
    形状。
    '''
    def backward(self, dout):       # dout(N,C,OH,OW)
        dout = dout.transpose(0, 2, 3, 1)       # (N,OH,OW,C)

        pool_size = self.pool_h * self.pool_w
        # 根据最大值的位置填充梯度矩阵
        dmax = np.zeros((dout.size, pool_size))   # dout.size是池化后的特征图元素个数，pool_size是每个池化窗口的大小
        #根据最大值位置填充梯度矩阵
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()  # 花式索引赋值

        # 将dmax重塑为与dout相同的形状，但额外增加了一个维度来存储每个输出位置的梯度
        # 将dout的形状和(pool_size, )连接起来，形成一个新的形状元组。在 dout.shape 的基础上增加一个新的维度，
        # 其大小为 pool_size。(pool_size,)如果不加逗号，它不会被解释为元组，而是被解释为一个数学表达式    即(N,OH,OW,C,pool_size)
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1)     # N*OH*OW,C*pool_size

        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.pad, self.stride)

        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
         if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
         else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask




