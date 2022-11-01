import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('image.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    # plt.imshow(img, cmap='gray')
    # plt.show()
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
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize))
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            temp[i][j] = convolution_5(img, i, j)
    # plt.imshow(temp, cmap='gray')
    # plt.show()
    return temp

def convolution_1(img, i, j):
    fx = np.array([
                        [-1,0],
                        [0,1]
                    ])
    val = np.sum(np.multiply(img[i:i+2, j:j+2], fx))
    return val
    
def convolution_2(img, i, j):
    fy = np.array([
                        [0,-1],
                        [1,0]
                    ])
    val = np.sum(np.multiply(img[i:i+2, j:j+2], fy))
    return val

def convolution(img, i, j):
    vertical = convolution_1(img, i, j)
    horizontal = convolution_2(img, i, j)
    return np.sqrt(np.square(vertical) + np.square(horizontal))

def sobel2x2(img, img_size, ksize):
    img = padding(img, img_size, ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize + 1, img_size - ksize + 1))
    for i in range(img_size - ksize + 1):
        for j in range(img_size - ksize + 1):
            temp[i][j] = convolution(img, i, j)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp

def median_filter(img, img_size, ksize):
    img = padding(img, img_size, ksize)
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
    return temp

def magnify(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 15:
                img[i][j] = img[i][j] + 100 
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def print_image(p, img , filtered_ima):
    print(p)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if filtered_ima[i][j] > 10:
                img[i][j] = max(img[i][j] - filtered_ima[i][j], 30)
    plt.imshow(img, cmap='gray')
    plt.show()

img = read_image()
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
img4 = img.copy()
img_filtered = Gaussian(img1, len(img1), 5)
img_sharpen = sobel2x2(img_filtered , len(img2), 2)
img_sharpen = magnify(img_sharpen)
print_image("first filter", img3, img_sharpen)

