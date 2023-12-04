import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('cameraman.png')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def Guassian_noise(img, mean, sigma):
    for i in range(len(img)):
        for j in range(len(img)):
            # print(np.random.normal(mean, sigma, 1))
            img[i][j]  = img[i][j] + np.random.normal(mean, sigma, 1)

    plt.imshow(img, cmap='gray')
    plt.show()
    return img

img = read_image()
img_gaussian = Guassian_noise(img, 0, 20)