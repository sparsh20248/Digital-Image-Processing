import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('image.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    cv.imshow("original picture", img)
    return img

def padding(img, img_size, ksize):
    temp = np.zeros((img_size + ksize, img_size + ksize))
    for i in range(img_size):
        for j in range(img_size):
            temp[i+ksize//2][j+ksize//2] = img[i][j]
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

def Gaussian(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize),dtype=np.uint8)
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            temp[i][j] = convolution_5(img, i, j)
    cv.imshow("Gaussian smoothening", temp)
 
    return temp

def convolution_1(img, i, j):
    sharpen = np.array([
                        [0, 0, -1, 0, 0],
                        [0, -1, -2, -1, 0],
                        [-1, -2, 16, -2, -1],
                        [0, -1, -2, -1, 0],
                        [0, 0, -1, 0, 0],
                    ])
    val = np.sum(np.multiply(img[i:i+5, j:j+5], sharpen))
    return val

def Laplacian(img, img_size, ksize):
    img = padding(img, img_size, ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize, img_size - ksize), dtype=np.uint8)
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            temp[i][j] = convolution_1(img, i, j)
    cv.imshow("making Laplacian filter", temp)
    return temp

def print_image(p, img , filtered_ima):
    print(p)
    img -= filtered_ima
    cv.imshow(p, img)

img = read_image()
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()

img_smoothed = Gaussian(img1, len(img1), 5)
img_smoothed = Gaussian(img_smoothed, len(img1), 5)
img_sharpened = Laplacian(img_smoothed, len(img2), 5)
print_image("final Image", img3, img_sharpened)
# print_image('Sharpened image', img, img_sharpened)
cv.waitKey(0)       
cv.destroyAllWindows()