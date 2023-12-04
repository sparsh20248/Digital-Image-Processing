import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('./../butterfly.png')
    img = ImageOps.autocontrast(img)
    img = np.array(img)
    plt.imshow(img)
    plt.show()
    return img

def padding(img, img_size, ksize):
    temp = np.zeros((img_size + ksize, img_size + ksize,3))
    for i in range(img_size):
        for j in range(img_size):
            temp[i+ksize//2][j+ksize//2] = img[i][j]
    return temp

def median_filter(img, img_size, ksize):
    img = padding(img, len(img), 3)
    img_size = len(img)
    temp = np.zeros((img_size - ksize,img_size - ksize, 3))
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            score = []
            for _i in range(ksize):
                for _j in range(ksize):
                    score.append(img[i+_i][j+_j][0])
            score.sort()
            number = score[len(score)//2]
            print(number)
            temp[i][j] = [number,number,number]
    print(temp.shape)
    plt.imshow(temp)
    plt.show()


img = read_image()
img1 = img.copy()
img2 = img.copy()
print(len(img1))
median_filter(img1, len(img1), 3)
# median_filter(img2, len(img2), 5)

