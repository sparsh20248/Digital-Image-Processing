import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm
def read_image():
    img = Image.open('image.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    cv.imshow("original picture", img)
    return img

def padding(img, img_size, ksize):
    temp = np.zeros((img_size + ksize, img_size + ksize))
    for i in range(img_size):
        for j in range(img_size):
            temp[i+ksize//2][j+ksize//2] = img[i][j]
    return temp

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
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize), dtype = np.uint8)
    for i in range(img_size - ksize ):
        for j in range(img_size - ksize ):
            temp[i][j] = convolution_5(img, i, j)
    # cv.imshow("Gaussian smoothening", temp)
    return temp

def mean_filter(img, img_size, ksize):
    img = padding(img, img_size, ksize)
    img_size = len(img)
    temp = np.zeros((img_size - ksize , img_size - ksize ))
    for i in range(img_size - ksize ):
        for j in range(img_size - ksize):
            score = 0
            for _i in range(ksize):
                for _j in range(ksize):
                    score += img[i+_i][j+_j]
            temp[i][j] = score/(ksize*ksize)
    return temp

def median_filter(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros((img_size - ksize ,img_size - ksize))
    for i in range(img_size - ksize):
        for j in range(img_size - ksize):
            score = []
            for _i in range(ksize):
                for _j in range(ksize):
                    score.append(img[i+_i][j+_j])
            score.sort()
            number = score[len(score)//2]   
            temp[i][j] = number
    return temp

def convolution_1(img, i, j):
    sharpen = np.array([
                        [0, 0, 3, 2, 2, 2, 3, 0, 0],
                        [0, 2, 3, 5, 5, 5, 3, 2, 0],
                        [3, 3, 5, 3, 0, 3, 5, 3, 3],
                        [2, 5, 3, -12, -23, -12, 3, 5, 2],
                        [2, 5, 0, -23, -40, -23, 0, 5, 2],
                        [2, 5, 3, -12, -23, -12, 3, 5, 2],
                        [3, 3, 5, 3, 0, 3, 5, 3, 3],
                        [0, 2, 3, 5, 5, 5, 3, 2, 0],
                        [0, 0, 3, 2, 2, 2, 3, 0, 0],
                    ])
    val = np.sum(np.multiply(img[i:i+9, j:j+9], sharpen))
    return val

def Laplacian(img, img_size, ksize):
    img = padding(img, img_size, ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize + 1, img_size - ksize + 1))
    for i in range(img_size - ksize + 1):
        for j in range(img_size - ksize + 1):
            temp[i][j] = convolution_1(img, i, j)
    # cv.imshow("making Laplacian filter", temp)
    return temp

def negative(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = 255 - img[i][j]
    return img

def thresholding(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] < 20:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img

def print_image(p, img , filtered_ima):
    print(p)
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = img[i][j] + (255 - filtered_ima[i][j])
    cv.imshow("final image", img)

img = read_image()
img1 = img.copy()
img2 = img.copy()

img_median = Gaussian(img1, len(img2), 5)
for i in range(2):
    img_median = mean_filter(img_median, len(img2), 3)
    img_median = median_filter(img_median, len(img2), 5)

img_log = Laplacian(img_median, len(img2), 9)
cv.imshow("Laplacian filter after 1", img_log)
# img_median = median_filter(img_log, len(img2), 3)
# for i in range(1):
#     img_median = median_filter(img_median, len(img2), 3)
    
# cv.imshow("Laplacian filter after 2", img_median)
img_log = Laplacian(img_log, len(img2), 9)
cv.imshow("Laplacian filter after 3", img_log)



cv.waitKey(0)       
cv.destroyAllWindows()