from dataset.mnist import load_mnist
#from SimpleConvNet import SimpleConvNet
from deep_convnet import DeepConvNet
import numpy as np
import matplotlib.pyplot as plt
import pickle

#network = SimpleConvNet(input_dim=(1, 28, 28), conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
#                        hidden_size=100, output_size=10, weight_init_std=0.01)

network = DeepConvNet(input_dim=(1, 28, 28),
                      conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                      conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                      conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                      conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                      conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                      conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                      hidden_size=50, output_size=10)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)

#定义超参数
iters_num = 10000
train_size = x_train.shape[0]  #60000
batch_size = 100
learning_rate = 0.1

#记录数据
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = train_size / batch_size  # 60000/100  600

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #误差反向传播法求梯度
    grad = network.gradient(x_batch, t_batch)

    #更新权重参数
    for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #每个epoch记录一次训练精度 测试精度   10000%600=16
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)  #?
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(train_acc, "；", test_acc)

# 将权重保存到文件中
with open('weights1.pkl', 'wb') as f:
    pickle.dump(network.params, f)

#print(network.params)



# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))        #x为一个数组，
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

lo = np.arange(iters_num)
plt.plot(lo, train_loss_list, label='loss')
plt.show()

