import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image():
    img = Image.open('clock.tiff')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def FFT(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    return fft_shift

def Low_pass_filter(img, fft, threshold):
    n = len(img)
    temp = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            xx = i - n//2
            yy = j - n//2
            dist = xx * xx + yy * yy
            if(dist < threshold): temp[i][j] = 1
    return temp

def convolve(fft, H):
    final_shift = fft * H
    final = np.fft.ifftshift(final_shift)
    final = np.fft.ifft2(final)
    final = np.abs(final)
    plt.imshow(final, cmap='gray')
    plt.axis('off')
    plt.show()
    return final

img = read_image()
fft = FFT(img)
H = Low_pass_filter(img, fft, 150)
convolve(fft, H)