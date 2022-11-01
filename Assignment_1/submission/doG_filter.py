from asyncore import read
from random import gauss
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
                        [1, 2, 1],  
                    ])
    val = np.sum(np.multiply(img[i:i+3, j:j+3], sharpen)) // 16
    return val

def Gaussian_3(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize),dtype=np.uint8)
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            temp[i][j] = convolution_3(img, i, j)
    cv.imshow("Gaussian smoothening", temp)
    return temp

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

def Gaussian_5(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize),dtype=np.uint8)
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            temp[i][j] = convolution_5(img, i, j)
    cv.imshow("Gaussian smoothening", temp)
    return temp

def print_image(p, img , filtered_ima):
    print(p)
    img -= filtered_ima
    cv.imshow(p, img)

img = read_image()
img1 = img.copy()
img2 = img.copy()

gaussian_3 = Gaussian_3(img1, len(img1), 3)
gaussian_5 = Gaussian_5(img2, len(img2), 5)
print(gaussian_3.shape, gaussian_5.shape)
output = gaussian_3 - gaussian_5
cv.imshow("mask = ", output)
print_image("final answer", img, output)

cv.waitKey(0)       
cv.destroyAllWindows()