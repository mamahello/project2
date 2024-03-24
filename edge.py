import cv2
import numpy as np

# 读取图像
image = cv2.imread('C:\project2\img.png', cv2.IMREAD_GRAYSCALE)

# 应用高斯滤波以减少噪声
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 使用Canny边缘检测算法
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# 显示原图和边缘检测结果
cv2.imshow('Original Image', image)
cv2.imshow('Edge Image', edges)

# 等待键盘输入，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()