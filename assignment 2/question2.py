import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('cameraman.png')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    temp = np.zeros((len(img) - 1, len(img) - 1))
    for i in range(512):
        for j in range(512):
            temp[i][j] = img[i][j]
    img = temp
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def interpolate(img, times):
    img_size = len(img)
    temp = np.zeros((img_size // times, img_size // times))
    print(temp.shape)
    for i in range(0, img_size, times): 
        for j in range(0, img_size, times):
            temp[i//times][j//times] = img[i][j]
    print_image(temp)
    return temp        

def nearest_neighbour(img, times):
    img_size = len(img)
    temp = np.zeros((img_size * times, img_size * times))
    for i in range(img_size * times):
        for j in range(img_size * times):
            temp[i][j] = img[i // times][j // times]
    print_image(temp)
    return temp
    
def print_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def PNSR(img, img_):
    img = np.array(img)
    img_ = np.array(img_)
    mse = np.mean((img - img_)**2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

img = read_image()
img_reduced= interpolate(img.copy(), 4)
img_enlarged = nearest_neighbour(img_reduced.copy(), 4)
print(PNSR(img, img_enlarged))

