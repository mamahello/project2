import numpy as np

def step_function(x):       #阶跃函数
    y=x>0
    return y.astype(np.int)  #astype转换类型

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def identity_function(x):
    return x


""""
softmax溢出问题
def softmax(x):
    exp_a=np.exp(x)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return   
    
"""
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)           #axis=0 每列的最大值
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

""""
损失函数

1.均方误差
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

2.交叉熵误差
def cross_entropy_error(y,t):
    delta=1e-7          #防止log0 导致后续计算无法进行
    return -np.sum(t*np.log(y+delta))

"""


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        '''
        mini_batch版交叉熵误差
        假设数据有n个，算出的交叉熵误差求和后要除以n,平均化（与训练数据的数量无关）
        判断维数，将单个数据和批量数据处理成同样的形式，
        方便后面shape[0]取出batch_size  
        如果y.ndim等于1，说明是单个数据的情况，此时batch_size应该为1，但是batch_size = y.shape[0]，
        y.shape[0]得到的值是输出神经元的个数而不是1，因此需要特殊处理一下
        '''

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引，此情况下t是一个数组（正确解的标签为1，别的全为0）
    if t.size == y.size:
        t = t.argmax(axis=1)     #t为一个数组，把one-hot改为非one-hot标签

    batch_size = y.shape[0]
    temple=np.arange(batch_size)   #temple为 0 - batch_size-1 的数组
    return -np.sum(np.log(y[temple, t] + 1e-7)) / batch_size     #1e-7  防止log（0）的情况
    # y[temple,t]  花式索引


#数值微分法求梯度
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    # np.nditer 迭代器 迭代访问数组
    # Flags参数用于指定迭代器的行为。multi_index是这些标志之一，它告诉nditer在迭代过程中追踪每个元素的多维索引。
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad




def im2col(input_data, filter_h, filter_w, stride = 1, pad = 0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], mode="constant",constant_values=0)

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):      ## 遍历卷积核的高度
        y_max = y + stride*out_h        ## 计算卷积核y位置对应的特征图的最大y坐标
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

            # 相当于对于卷积核的每一个确定的位置（x,y）,把与该点进行运算的在特征图对应元素全部找出
            # img中切出去的数组刚好为N,C,out_h,ouy_w

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    # 把数据改为N,out_h,out_w,C,filter_h,filter_w 再改为形状 （N*out_h*out_w，C*filter_h*filter_w)
    # 再把卷积核展开为C*filter_h*filter_w，即可实现卷积运算（点积即可）
    return col



def col2im(col, input_shape, filter_h, filter_w, pad=0, stride = 1):
    """
    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    #col二维数组 N*OH*OW,C*FH*FW
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)

    img = np.zeros((N, C, H+2*pad+stride-1,W+2*pad+stride-1))
    # 这个数组的大小是为了确保能够容纳卷积操作后的所有输出值，包括由于步长和填充导致的额外空间。多出了stride-1

    for y in range(filter_h):      #遍历卷积核的每个位置y,x
        y_max = y + stride*out_h        #在img上卷积核y位置对应的最大位置
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :,y ,x ,: ,:]
    """ 
      遍历卷积核
      y:y_max:stride 和 x:x_max:stride 是切片操作，用于选取 img 中特定的区域。y:y_max:stride 表示从 y 开始，每隔 stride 个元素，
      直到（但不包括）y_max。同理，x:x_max:stride 表示从 x 开始，每隔 stride 个元素，直到（但不包括）x_max。只选取 img 中所有对应于
      当前卷积核(x,y)位置并与之运算的区域。(找出img上卷积核的x,y位置对应的所有的点)
      col[:, :, y, x, :, :]选取 col 中对应(y, x) 的所有元素。这里的 y 和 x 是卷积核在输出特征图上的位置。
      """

    return img[:, :, pad:pad+H, pad:pad+W]









