import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
sigma = 4
def read_image():
    img = Image.open('./../ruler.512_2.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def mask(img):
    img_size = len(img)
    mask = np.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            mask[i][j] = np.exp(-(i*i + j*j) / 2*sigma)/ (np.sqrt(2)*np.pi*sigma)
    mask = mask[1:,1:]
    return mask

def convolute(img,mask):
    image = img*mask
    print(image)
    plt.imshow(image, cmap='gray')
    plt.show()
    
img = read_image()
mask = mask(img)
convolute(img, mask)