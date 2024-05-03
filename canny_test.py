import cv2
import numpy as np
image_ori = r"./imgs/003.jpg"
gray = cv2.imread(image_ori,0)

# 使用 Canny 边缘检测算法
edges = cv2.Canny(gray, 100, 200, apertureSize=3)
cv2.imshow('Corrected Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()