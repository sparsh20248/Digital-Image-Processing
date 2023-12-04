import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('cameraman.png')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    temp = np.zeros((len(img) - 1, len(img) - 1), dtype = np.uint8)
    for i in range(512):
        for j in range(512):
            temp[i][j] = img[i][j]
    img = temp
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def interpolate(img, times):
    img_size = len(img)
    temp = np.zeros((img_size // times, img_size // times), dtype = np.uint8)
    print(temp.shape)
    for i in range(0, img_size, times): 
        for j in range(0, img_size, times):
            temp[i//times][j//times] = img[i][j]
    print_image(temp)
    return temp        



def nearest_neighbour(img, times):
    img_size = len(img)
    temp = np.zeros((img_size * times, img_size * times), dtype = np.uint8)
    for i in range(img_size):
        for j in range(img_size):
            temp[i*times][j*times] = img[i][j]
            if(i*times + 1 >= 2 * img_size): continue
            else: temp[i*times + 1][j*times] += img[i][j]//2
            if(i*times - 1 < 0): continue
            else: temp[i*times - 1][j*times] += img[i][j]//2
            if(j*times + 1 >= 2 * img_size): continue
            else: temp[i*times][j*times+1] += img[i][j]//2
            if(j*times - 1 <0): continue
            else: temp[i*times][j*times-1] += img[i][j]//2
            if(i*times + 1 >= 2 * img_size or j*times + 1 >= 2 * img_size): continue
            else: temp[i*times + 1][j*times + 1] += img[i][j]//4
            if(i*times + 1 >= 2 * img_size or j*times - 1 <0): continue
            else: temp[i*times + 1][j*times - 1] += img[i][j]//4
            if(i*times - 1 < 0 or j*times + 1 >= 2 * img_size): continue
            else: temp[i*times - 1][j*times + 1] += img[i][j]//4
            if(i*times - 1 < 0 or j*times - 1 <0): continue
            else: temp[i*times - 1][j*times - 1] += img[i][j]//4 
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
    print(psnr)
    
    return psnr

img = read_image()
print(img.shape)
img_reduced= interpolate(img.copy(), 4)
img_enlarged = nearest_neighbour(img_reduced.copy(), 2)
print(img_enlarged.shape)
img_enlarged = nearest_neighbour(img_enlarged.copy(), 2)
print(img_enlarged.shape)
PNSR(img, img_enlarged)


