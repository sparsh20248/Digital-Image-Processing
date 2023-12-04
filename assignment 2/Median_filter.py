import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('barbara_gray.bmp')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def PNSR(img, img_):
    img = np.array(img)
    img_ = np.array(img_)
    mse = np.mean((img - img_)**2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    print(psnr)
    return psnr

def padding(img, img_size, ksize):
    temp = np.zeros((img_size + ksize, img_size + ksize))
    for i in range(img_size):
        for j in range(img_size):
            temp[i+ksize//2][j+ksize//2] = img[i][j]
    return temp

def median_filter(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros((img_size - ksize,img_size - ksize))
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            score = []
            for _i in range(ksize):
                for _j in range(ksize):
                    score.append(img[i+_i][j+_j])
            score.sort()
            number = score[len(score)//2]
            temp[i][j] = number
    print(temp.shape)
    plt.imshow(temp, cmap='gray')
    plt.show()


img = read_image()
img1 = img.copy()
img2 = img.copy()
median_filter(img1, len(img1), 3)
median_filter(img2, len(img2), 5)