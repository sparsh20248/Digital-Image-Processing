import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
#first we need to take input of the image
def read_image():
    img = Image.open('clock.tiff')
    #converting greyscale and making it numpy array.
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    # print(img.shape)
    return img

def nearest_neighbour(img):
    ## creating a new image 2*len and 2*breath => area becomes 4 times
    temp = np.zeros((len(img)*2, len(img)*2))
    for i in range(len(img)):
        for j in range(len(img)):
            #for each img[i][j] it will expand to 4 values => (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            temp[2*i][2*j] = img[i][j]
            temp[2*i+1][2*j] = img[i][j]
            temp[2*i][2*j+1] = img[i][j]
            temp[2*i+1][2*j+1] = img[i][j]
    # print(temp)
    plt.imshow(temp, cmap='gray')
    plt.show()

img = read_image()
nearest_neighbour(img)