import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('image.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def padding(img, img_size, ksize):
    temp = np.zeros((img_size + ksize, img_size + ksize))
    for i in range(img_size):
        for j in range(img_size):
            temp[i+ksize//2][j+ksize//2] = img[i][j]
    return temp

def convolution_3(img, i, j):
    sharpen = np.array([
                        [1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]
                    ])
    val = np.sum(np.multiply(img[i:i+3, j:j+3], sharpen)) // 16
    return val

def convolution_5(img, i, j):
    sharpen = np.array([
                        [1, 4, 7, 4, 1],
                        [4, 16, 26, 16, 4],
                        [7, 26, 41, 26, 7],
                        [4, 16, 26, 16, 4],
                        [1, 4, 7, 4, 1]
                    ])
    val = np.sum(np.multiply(img[i:i+5, j:j+5], sharpen)) // 273
    return val

def Gaussian(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize))
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            if ksize == 3:
                temp[i][j] = convolution_3(img, i, j)
            else:
                temp[i][j] = convolution_5(img, i, j)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp

img = read_image()
img_sharpen = Gaussian(img, len(img), 5)
print(img_sharpen.shape)