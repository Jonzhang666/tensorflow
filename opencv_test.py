#导入cv模块
import numpy as np
import cv2 as cv
#读取图像，支持 bmp、jpg、png、tiff 等常用格式
img = cv.imread("D:/pythonworkplace/1/test.jpg")
#创建窗口并显示图像
cv.namedWindow('Image',cv.WINDOW_AUTOSIZE)
cv.imshow('Image',img)
cv.waitKey(0)
#释放窗口
cv2.destroyAllWindows() 
