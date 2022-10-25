from statistics import mean
import cv2 as cv
import numpy as np
import copy

def mean_filter(img, ksize):
    temp = np.zeros((513 - ksize, 513 - ksize, 3), dtype = np.uint8)
    for i in range(513 - ksize):
        for j in range(513 - ksize):
            score = 0
            for _i in range(ksize):
                for _j in range(ksize):
                    score += img[i+_i][j+_j][0]
            temp[i][j] = [score/(ksize*ksize), score/(ksize*ksize), score/(ksize*ksize)]
    cv.imshow(f"Mean Image {ksize} x {ksize}", temp)


img = cv.imread('image.tiff')
cv.imshow("IMAGE" ,img)
mean_filter(img, 3)
mean_filter(img, 5)

cv.waitKey(0)       
cv.destroyAllWindows()
