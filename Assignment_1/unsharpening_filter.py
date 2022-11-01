import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import cv2 as cv

def read_image():
    img = Image.open('image.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    return img

def padding(img, img_size, ksize):
    temp = np.zeros((img_size + ksize, img_size + ksize), dtype=np.uint8)
    for i in range(img_size):
        for j in range(img_size):
            temp[i+ksize//2][j+ksize//2] = img[i][j]
    return temp

def mean_filter(img, img_size, ksize):
    img = padding(img, img_size, ksize)
    img_size = len(img)
    temp = np.zeros((img_size - ksize, img_size - ksize),   dtype=np.uint8)
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            score = 0
            for _i in range(ksize):
                for _j in range(ksize):
                    score += img[i+_i][j+_j]
            temp[i][j] = score/(ksize*ksize)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp


img = read_image()
img1 = img.copy()

img2 = mean_filter(img1, len(img1), 5)
img2 = mean_filter(img2, len(img1), 5)
img2 = mean_filter(img2, len(img1), 5)
mask = img - img2
cv.imshow('mask', mask)
output = img + mask
cv.imshow('final Image', output)

cv.waitKey(0)       
cv.destroyAllWindows()
# mean_filter(img2, len(img2), 5)
