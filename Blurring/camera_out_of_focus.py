import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
c = 4
def read_image():
    img = Image.open('./../submission/clock.tiff')
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

def convolution_3(img, i, j):
    r = 5
    answer = 0
    for ii in range(r):
        for jj in range(r):
            if(ii==2 and jj==2): 
                answer+=img[i][j]
                continue
            x = i - 2 + ii
            y = i - 2 + jj
            # print(x,y,end=' ')
            if(x<0 or x>=len(img) or y<0 or y>=len(img)): continue
            elif(ii*ii + jj*jj <= r*r): 
                temp = img[x][y]/(c*np.pi*((abs(ii-2))*(abs(ii-2)) + (abs(jj-2))*(abs(jj-2))))
                # print(temp)
                answer+= temp

    return answer

def convolution_5(img, i, j):
    sharpen = np.array([
                        [1, 4, 7, 4, 1],
                        [4, 16, 26, 16, 4],
                        [7, 26, 41, 26, 7],
                        [4, 16, 26, 16, 4],
                        [1, 4, 7, 4, 1]
                    ])
    val = np.sum(np.multiply(img[i:i+5, j:j+5], sharpen)) // 273
    return val

def Gaussian(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize))
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            temp[i][j] = convolution_3(img, i, j)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp

img = read_image()
img_sharpen = Gaussian(img, len(img), 3)
print(img_sharpen)
# img_sharpen_2 = Gaussian(img_sharpen, len(img_sharpen), 5)
# print(img_sharpen.shape)