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
    for i in range(len(img)-1):
        for j in range(len(img)-1):
            temp[2*i][2*j] = img[i][j]
            temp[2*i+1][2*j] = (img[i][j])/2+(img[i+1][j])/2
            temp[2*i][2*j+1] = (img[i][j])/2+(img[i][j+1])/2
            temp[2*i+1][2*j+1] = (img[i][j])/2+(img[i+1][j+1])/2
    for i in range(2*len(img)-1):
        for j in range(2*len(img)-1):
            if(temp[i][j] == 0): print(temp[i][j], end=' ')
    # for i in range(len(img)-1):
    #     for j in range(len(img)-1):
    #         if(temp[2*i+1][2*j+1] == 0):
    #             temp[2*i+1][2*j+1] = (temp[2*i+1][2*j] + temp[2*i+1][2*j+2])/2
        # temp[2*i][len(img)*2-1] = img[i][len(img)-1]
        # temp[2*i+1][len(img)*2-1] = (img[i][len(img)-1]+img[i+1][len(img)-1])//2
    print(temp)
    plt.imshow(temp, cmap='gray')
    plt.show()


img = read_image()
print(img)
nearest_neighbour(img)