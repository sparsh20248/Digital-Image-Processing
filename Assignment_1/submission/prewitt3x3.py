from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('image.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    # plt.imshow(img, cmap='gray')
    # plt.show()
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
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize))
    for i in tqdm(range(img_size - ksize), "smoothening"):
        for j in range(img_size - ksize):
            temp[i][j] = convolution_5(img, i, j)
    # plt.imshow(temp, cmap='gray')
    # plt.show()
    return temp

def convolution_1(img, i, j):
    fx = np.array([
                        [-1, -1 , -1],
                        [0, 0, 0],
                        [1, 1, 1]
                    ])
    val = np.sum(np.multiply(img[i:i+3, j:j+3], fx))
    return val
    
def convolution_2(img, i, j):
    fy = np.array([
                        [-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]
                    ])
    val = np.sum(np.multiply(img[i:i+3, j:j+3], fy))
    return val

def convolution(img, i, j):
    vertical = convolution_1(img, i, j)
    horizontal = convolution_2(img, i, j)
    return np.sqrt(np.square(vertical) + np.square(horizontal))

def prewitt3x3(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize + 1, img_size - ksize + 1))
    for i in tqdm(range(img_size - ksize + 1), "filtering"):
        for j in range(img_size - ksize + 1):
            temp[i][j] = convolution(img, i, j)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp

def magnify(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 50:
                img[i][j] = img[i][j] + 150 
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def print_image(p, img , filtered_ima):
    print(p)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if filtered_ima[i][j] > 50:
                img[i][j] = max(img[i][j] - filtered_ima[i][j], 30)
    plt.imshow(img, cmap='gray')
    plt.show()

img = read_image()
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()

img_smoothened = Gaussian(img1, len(img1), 5)
img_s = img_smoothened.copy()
img_sharpened = prewitt3x3(img_smoothened, len(img2), 3)
img_sharpened = magnify(img_sharpened)
print_image("Sharpened image", img3, img_sharpened)