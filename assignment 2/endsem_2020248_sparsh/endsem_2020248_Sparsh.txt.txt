from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# this function is used to read the image as input
def read_image():
    img = Image.open('final_test.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

#this is very important as the image will loose its dimension why using the filter. This will ensure a padding is
#so that after the image is convolved the original dimension is maintained
def padding(img, img_size, ksize):
    temp = np.zeros((img_size + ksize, img_size + ksize))
    for i in range(img_size):
        for j in range(img_size):
            temp[i+ksize//2][j+ksize//2] = img[i][j]
    return temp

#this is he gaussian filter that will smoothen the image. This is to remove any noise in the iamge, in case there is any.
#this is an optional step, yet recommended as we need to ensure that noise is not detected as edges.
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
#this is the Gaussian filter funciton that will call the above filter.
def Gaussian(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize , img_size - ksize))
    for i in tqdm(range(img_size - ksize), "smoothening"):
        for j in range(img_size - ksize):
            temp[i][j] = convolution_5(img, i, j)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp

#this is the horizontal edge detection filter. How is works is simple, it has -1, -1, -1 on the top 
#and 0, 0, 0 in the middle and 1, 1, 1 in the bottom. Now if there is an edge then lets say we are changeing from 0 
#to 255 then we will have a bigger value for the pixel.
#in case there is no edge then all values will be around x. so -x -x -x + 0 + 0 + 0 + x + x + x = (approx). so we will
#have a black vlue for this pixel. 
def convolution_horizontal(img, i, j):
    fx = np.array([
                        [-1, -1 , -1],
                        [0, 0, 0],
                        [1, 1, 1]
                    ])
    val = np.sum(np.multiply(img[i:i+3, j:j+3], fx))
    return val

# same theory in vertical direction
def convolution_vertical(img, i, j):
    fy = np.array([
                        [-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]
                    ])
    val = np.sum(np.multiply(img[i:i+3, j:j+3], fy))
    return val

def combine(img_vertical, img_horizontal):
    edges = np.sqrt(np.square(img_vertical) + np.square(img_horizontal))
    plt.imshow(edges, cmap='gray')
    plt.show()
    return edges

def prewitt3x3_vertical(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize + 1, img_size - ksize + 1))
    for i in tqdm(range(img_size - ksize + 1), "vertical edges"):
        for j in range(img_size - ksize + 1):
            temp[i][j] = convolution_vertical(img, i, j)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp

def prewitt3x3_horizontal(img, img_size, ksize):
    img = padding(img, len(img), ksize)
    img_size = len(img)
    temp = np.zeros(shape = (img_size - ksize + 1, img_size - ksize + 1))
    for i in tqdm(range(img_size - ksize + 1), "horizontal edges"):
        for j in range(img_size - ksize + 1):
            temp[i][j] = convolution_horizontal(img, i, j)
    plt.imshow(temp, cmap='gray')
    plt.show()
    return temp

img = read_image()
# this step is completely optional as we dont always need to smoothen the image. Smoothening is done to remove any noise,
# as noises can be detected as edges.
# img = Gaussian(img.copy(), len(img), 5) (commenting as not requied in the question)
#prewitt 3x3 vertical will find the vertical edges in the image. these edges can be seen as ----- in the image
img_horizontal = prewitt3x3_vertical(img.copy(), len(img), 3)
#prewitt 3x3 horizontal will find the horizontal edges in the image. these edges can be seen as ||||| in the image
img_vertical = prewitt3x3_horizontal(img.copy(), len(img), 3)
# finally we will combine our vertical and horizontal edges to get the final image with all the edges
edges = combine(img_vertical, img_horizontal)