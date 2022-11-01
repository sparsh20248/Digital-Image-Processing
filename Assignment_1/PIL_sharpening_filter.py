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

def convolution_1(img, i, j):
    sharpen = np.array([
                        [-2, -2, -2],
                        [-2, 32, -2],
                        [-2, -2, -2]
                    ])
    val = np.sum(np.multiply(img[i:i+3, j:j+3], sharpen))
    return val

def Gaussian(img, img_size, ksize):
    temp = np.zeros(shape = (img_size - ksize + 1, img_size - ksize + 1), dtype = np.uint8)
    for i in range(img_size - ksize + 1):
        for j in range(img_size - ksize + 1):
            temp[i][j] = convolution_1(img, i, j)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp

img = read_image()
img1 = padding(img, len(img), 3)
img_sharpen = Gaussian(img1, len(img1), 3)