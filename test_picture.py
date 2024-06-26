from deal_img import dealimg
from two_layer_net_2 import TwoLayerNet
import numpy as np

path = 'C:\\Users\\marenkun\\Desktop\\picture1\\8.jpg'
img_array = dealimg(path)
img_array = img_array.reshape(1, 784)   #affine层前向传播处理张量情况
my_network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
my_network.load_weights('weights.pkl')
prediction = my_network.predict(img_array)
a = np.argmax(prediction)
print("这张图片的数字是：", a)