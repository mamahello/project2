from deal_img import dealimg
from two_layer_net_2 import TwoLayerNet
import numpy as np

path = 'C:\\Users\\marenkun\\Desktop\\picture1\\data3.jpg'
img_array = dealimg(path)

my_network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
my_network.load_weights('weights.pkl')
prediction = my_network.predict(img_array)
a = np.argmax(prediction)
print("这张图片的数字是：", a)