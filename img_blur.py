import numpy as np
import cv2 as cv


# 均值滤波
def avg_blur(img):
    blur = cv.blur(img, (5, 5))
    return blur


# 高斯滤波
def gaussian_blur(img):
    blur = cv.GaussianBlur(img, (5, 5), 0)
    return blur


# 增加椒盐噪声
def add_sp_noise(img, sp_number):
    new_image = img
    row, col, channel = img.shape  # 获取行列,通道信息
    s = int(sp_number * img.size / channel)  # 根据sp_number确定椒盐噪声
    # 确定要扫椒盐的像素值
    change = np.concatenate((np.random.randint(0, row, size=(s, 1)), np.random.randint(0, col, size=(s, 1))), axis=1)
    for i in range(s):
        r = np.random.randint(0, 2)  # 确定撒椒（0）还是盐（1）
        for j in range(channel):
            new_image[change[i, 0], change[i, 1], j] = r
    return new_image


# 中值滤波
def med_blur(img, percent):
    median = cv.medianBlur(img, percent)
    return median


# Canny边缘检测
def canny(img, low_thresh, high_thresh):
    r1 = cv.Canny(img, low_thresh, high_thresh)
    return r1


def show_img(img_s, name_img):

    cv.imshow(name_img, img_s)
    cv.waitKey(0)
    # cv.destroyAllWindows()


if __name__ == '__main__':
    ori_img = cv.imread(r"D:\imgs\004.png", cv.IMREAD_GRAYSCALE)
    dst = gaussian_blur(ori_img)
    show_img(ori_img, "original")
    # show_img(dst, "blur_res")
    # add_noise_img = add_sp_noise(ori_img, 0.05)
    # show_img(add_noise_img, "noise")
    # med_blur = med_blur(add_noise_img, 5)
    # show_img(med_blur, "med_blur")
    canny_res = canny(ori_img, 128, 200)
    show_img(canny_res, "canny_blur")




