from two_layer_net_2 import TwoLayerNet
from PIL import Image
import numpy as np
import os, sys
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)

i = 9
img = x_test[i]
table = t_test[i]
'''
def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))    #image.fromarray()  把保存为numpy数组的图像数据转换为PIL用的数据对象 unit8 无符号
    pil_img.show()

img = img.reshape(28, 28)
img_show(img)
'''
img = img.reshape(784)

my_network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#加载权重
my_network.load_weights('weights.pkl')

prediction = my_network.predict(img)
a = np.argmax(prediction)
print("这张图片的数字是：", a, ",标签是：",table)
