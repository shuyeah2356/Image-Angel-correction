import cv2
import numpy as np


def correct_skew(img):
    """
    处理，计算图片倾斜角度，然后旋转图片，纠正图片
    :param img: 原图
    :return:
    """
    h = img.shape[0]
    w = img.shape[1]
    # 将图像转换为灰度图,检测边缘，不需要色彩信息（1、去除色彩干扰；2、降低需要处理的数据）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 Canny 边缘检测算法
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 280)
    # 绘画霍夫直线
    if lines is not None:
        for n, line in enumerate(lines):
            # 沿着左上角的原点，作目标直线的垂线得到长度和角度
            rho = line[0][0]
            theta = line[0][1]
            # if np.pi / 3 < theta < np.pi * (3 / 4):
            a = np.cos(theta)
            b = np.sin(theta)
            # 得到目标直线上的点
            x0 = a * rho
            y0 = b * rho
    
            # 延长直线的长度，保证在整幅图像上绘制直线
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
    
            # 连接两点画直线
            # print((x1, y1), (x2, y2))  # (-148, 993) (335, -947)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
            # ===============================CAB================================ #
            xDis = x2 - x1  # x的增量
            yDis = y2 - y1  # y的增量
            if (abs(xDis) > abs(yDis)):
                maxstep = abs(xDis)
            else:
                maxstep = abs(yDis)
            xUnitstep = xDis / maxstep  # x每步骤增量
            yUnitstep = yDis / maxstep  # y的每步增量
            x = x1
            y = y1
            average_gray = []
            for k in range(maxstep):
                x = x + xUnitstep
                y = y + yUnitstep
                # print("x: %d, y:%d" % (x, y))
                if 0 < x < h and 0 < y < w:
                    # print(img_gray[int(x), int(y)])
                    average_gray.append(img[int(x), int(y)])
            print('第{}霍夫直线的平均灰度值：'.format(n), np.mean(average_gray))  # 平均115，阴影的边界在125以上，堵料的边界在105左右
            # ================================================================== #
    
        print('直线的数量：', len(lines))
    else:
        print('直线的数量：', 0)
    
    # 可视化图像
    cv2.imshow('0', img)
    cv2.imshow('1', edges)
    cv2.waitKey(0)




def main():
    # 读取图像
    image = cv2.imread(r".\imgs\003.jpg")
    # 显示结果
    cv2.imshow('Origin Image', image)

    # 对图像进行倾斜纠正
    correct_skew(image)



if __name__ == '__main__':
    main()