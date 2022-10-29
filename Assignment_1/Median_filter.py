import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('ruler.512_2.tiff')
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

def median_filter(img, img_size, ksize):
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
img1 = padding(img, len(img), 3)
median_filter(img1, len(img1), 3)
img2 = padding(img, len(img), 5) 
median_filter(img2, len(img2), 5)

