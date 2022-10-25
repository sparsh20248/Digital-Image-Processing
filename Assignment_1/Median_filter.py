from statistics import mean
import cv2 as cv
import numpy as np
import copy

def median_filter(img, ksize):
    temp = np.zeros((513 - ksize, 513 - ksize, 3), dtype = np.uint8)
    for i in range(513 - ksize):
        for j in range(513 - ksize):
            score = []
            for _i in range(ksize):
                for _j in range(ksize):
                    score.append(img[i+_i][j+_j][0])
            number = score[len(score)//2]
            temp[i][j] = [number, number, number]
    cv.imshow(f"Median Image {ksize} x {ksize}", temp)


img = cv.imread('image.tiff')
cv.imshow("IMAGE" ,img)
median_filter(img, 3)
median_filter(img, 5)

cv.waitKey(0)       
cv.destroyAllWindows()
