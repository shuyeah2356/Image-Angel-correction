import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import Counter

# 读取图像
img = cv2.imread(r"D:\imgs\003.jpg",2)
img = 255 - img

print(img.shape)  # [204,547]

# 正变换：将xy坐标系中的点映射到极坐标中，记录映射函数经过的每一点
# 根据由直角坐标变换为极坐标，xcost+ysint=ρ
def hough_forward_conver(x, y, points):
	for t in range(0,360,2):
		r = int(x * np.cos(np.pi*t/180) + y * np.sin(np.pi*t/180))
		points.append([t,r])  # 直线经过的点放进去
	return points

# 反变换：根据极坐标系的坐标求xy坐标系的坐标
def hough_reverse_conver(y, t, r):
	x = int(- y * (np.sin(np.pi*t/180) / (np.cos(np.pi*t/180)+ 1e-4)) + r / (np.cos(np.pi*t/180)+1e-4))
	return x

# 霍夫正变换
points = []  # 存放变换后的直线经过的点
px, py = np.where(img == 255)  # 检测出直线上的点
for x, y in zip(px, py):
	points = hough_forward_conver(x,y,points)  # 霍夫变换，xy--->theta,rho
print(len(points))

# 画极坐标图
points = np.array(points)
print(points)
# theta, rho = points[:,0], points[:,1]
# ax = plt.subplot(111, projection='polar')
# ax.scatter(np.pi*theta/180, rho, c='b', alpha=0.5,linewidths=0.01)

# 霍夫空间的网格坐标系
hough_space = np.zeros([360, 3000])
for point in points:
    t, r = point[0], point[1] + 1000  # r可能为负，防止索引溢出     h
    hough_space[t,r] += 1

# 找出直线所在的点
line_points = np.where(hough_space >= 15)
print(len(line_points[0]))

# 霍夫逆变换求xy
mask = np.zeros_like(img)
for t,r in zip(line_points[0],line_points[1]):
     for y in range(img.shape[0]):
         x = hough_reverse_conver(y, t,r-1000)
         if x in range(1,img.shape[1]):
         	mask[y,x] += 1

plt.imshow(mask)
plt.imshow(img)
