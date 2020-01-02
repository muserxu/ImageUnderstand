import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('bookCover.jpg',0)          # queryImage
im2 = cv2.imread('im3.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(im1,None)
kp2, des2 = sift.detectAndCompute(im2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)


# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatches(im1,kp1,im2,kp2,matches,flags=2, outImg = None)

plt.imshow(img3),plt.show()