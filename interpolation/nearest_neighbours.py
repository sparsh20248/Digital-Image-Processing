import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('./../butterfly.png')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def nearest_neighbour(img):
    temp = np.zeros((len(img)*2, len(img)*2))
    for i in range(len(img)):
        for j in range(len(img)):
            temp[2*i][2*j] = img[i][j]
            temp[2*i+1][2*j] = img[i][j]
            temp[2*i][2*j+1] = img[i][j]
            temp[2*i+1][2*j+1] = img[i][j]
    print(temp)
    plt.imshow(temp, cmap='gray')
    plt.show()


img = read_image()
print(img)
nearest_neighbour(img)