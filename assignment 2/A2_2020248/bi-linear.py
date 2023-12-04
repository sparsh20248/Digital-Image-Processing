import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

#this function read the image and convert it to grayscale
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

#this function reduce the image size from 512x512 to 128x128
def reduce(img, times):
    img_size = len(img)
    temp = np.zeros((img_size // times, img_size // times), dtype = np.uint8)
    print(temp.shape)
    for i in range(0, img_size, times): 
        for j in range(0, img_size, times):
            temp[i//times][j//times] = img[i][j]
    print_image(temp)
    return temp        

#this function is used to make the kernal. 1-D kernal is made using the conditions given in the lecture notes and futher taking outer product we get the 
#2-D kernal. Here 1-d kernal is [0, 1/4, 2/4, 3/4 , 1, 3/4, 2/4, 1/4, 0]. 
def make_kernal():
    k = [0, 1/4, 2/4, 3/4 , 1, 3/4, 2/4, 1/4, 0]
    kernal = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            kernal[i][j] = k[i] * k[j]
    return kernal


#this function acts like a convolution step. That is it fills all the values of new pixels that will be affected by our image. The weight here is defined 
#by the kernal. if the value is 200 and kernel is 0.5 then the new pixel value will be 100.
def convolve(k, temp, img, times, x, y):
    for i in range(0, 9):
        for j in range(0, 9):
            x_ = x * times + i - 4
            y_ = y * times + j - 4
            if(x_ < 0 or x_ >= 512 or y_ < 0 or y_>= 512):
                continue
            temp[x_][y_] += img[x][y] * k[i][j]

#this the bi-linear function. All tyhe functions will be the same, going to each pixel in the small image and changing all the effecting pixels in the new image.
def bilinear(img, times, kernel):
    img_size = len(img)
    temp = np.zeros((img_size * times, img_size * times), dtype = np.uint8)
    for i in range(img_size):
        for j in range(img_size):
            convolve(kernel, temp , img, times, i, j)
    print_image(temp)
    return temp
    
#this function is used to print the image
def print_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()

#this function is used to calculate the PNSR
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

#this is the main funciton is used to make the calls to the functions.
# steps
# 1) we read the image
# 2) we make the kernal
# 3) we reduce the image size
# 4) we enlarge the image using our kernel
# 5) we calculate the PNSR
img = read_image()
kernel = make_kernal()
img_reduced= reduce(img.copy(), 4)
img_enlarged = bilinear(img_reduced.copy(), 4, kernel)
PNSR(img, img_enlarged)


