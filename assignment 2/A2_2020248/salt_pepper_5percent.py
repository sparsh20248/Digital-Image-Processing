import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import math
#first we need to take input of the image

def salt_and_pepper(percent, img):
    salt = percent/2
    pepper = 1 - percent/2
    for i in range(len(img)):
        for j in range(len(img)):
            r = np.random.random()
            if r < salt:
                img[i][j] = 255
            elif r > pepper:
                img[i][j] = 0
            else: 
                img[i][j] = img[i][j]
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def PNSR(img, img_):
    mse = np.mean((img - img_)**2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    print(psnr)
    return psnr

def read_image():
    img = Image.open('barbara_gray.bmp')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def padding(img, img_size, ksize):
    temp = np.zeros((img_size + ksize, img_size + ksize), dtype = np.uint8)
    for i in range(img_size):
        for j in range(img_size):
            temp[i+ksize//2][j+ksize//2] = img[i][j]
    return temp

def median_filter(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros((img_size - ksize,img_size - ksize), dtype = np.uint8)
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
    
img = read_image()
img_5 = salt_and_pepper(0.05, img.copy())
denoised_image = median_filter(img_5.copy(), len(img_5), 2)
PNSR(img, denoised_image)
denoised_image = median_filter(img_5.copy(), len(img_5), 3)
PNSR(img, denoised_image)