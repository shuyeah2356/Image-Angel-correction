"""
简单图像倾斜校正处理效果
    1、首先将图像转换为灰度图，并使用 Canny 边缘检测算法检测图像边缘。
    2、然后，我们使用霍夫变换检测图像中的直线，并计算直线的角度。
    3、接着，我们计算直线角度的中位数，并根据该角度对图像进行旋转。
    4、最后，我们显示了纠正后的图像。
"""

import cv2
import numpy as np


def correct_skew(img):
    """
    处理，计算图片倾斜角度，然后旋转图片，纠正图片
    :param img: 原图
    :return:
    """
    # 将图像转换为灰度图,检测边缘，不需要色彩信息（1、去除色彩干扰；2、降低需要处理的数据）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 Canny 边缘检测算法
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # 检查是否检测到直线
    if lines is None:
        print("No lines detected. Using default angle.")
        return img

    # 计算直线的角度
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        angles.append(angle)

    # 计算直线角度的中位数
    median_angle = np.median(angles)-90
    print(median_angle, '旋转角度的中位数')

    # 对图像进行旋转
    rotated = rotate_image(img, median_angle)

    return rotated


def rotate_image(img, angle):
    """
    旋转纠正图片
    :param img:
    :param angle:
    :return:
    """
    # 获取图像的中心点坐标
    height, width = img.shape[:2]
    center = (width / 2, height / 2)

    # 计算旋转矩阵
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 进行图像旋转
    rotated = cv2.warpAffine(img, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def main():
    # 读取图像
    image = cv2.imread(r"D:\imgs\003.jpg")
    # 显示结果
    cv2.imshow('Origin Image', image)

    # 对图像进行倾斜纠正
    corrected_image = correct_skew(image)

    # 显示结果
    cv2.imshow('Corrected Image', corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()