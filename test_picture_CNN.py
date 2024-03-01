from deal_img import dealimg
from cnn import SimpleConvNet
import numpy as np

path = 'C:\\Users\\marenkun\\Desktop\\picture1\\8.jpg'
img_array = dealimg(path)
img_array = img_array.reshape(1, 1, 28, 28)

my_network = SimpleConvNet(input_dim=(1, 28, 28), conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
my_network.load_params(file_name='weights1.pkl')
prediction = my_network.predict(img_array)
a = np.argmax(prediction)
print("这张图片的数字是：", a)