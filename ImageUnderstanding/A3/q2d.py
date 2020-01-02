import numpy as np 
import random
import cv2 as cv2
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.filters as nd_filters
# from astropy.modeling.models import Gaussian2D
from scipy.ndimage import gaussian_filter
import math
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import LineModelND, ransac
from skimage.transform import AffineTransform, warp, ProjectiveTransform


src = cv2.imread('anotherBookCover.jpg',0)
dst = cv2.imread('im3.jpg')
cols, rows= src.shape


c = np.array([[0, 0], [rows, 0], [rows-1,cols-1], [0,cols-1]])


plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.xlim(300, 600)
plt.ylim(800, 400)

x = plt.ginput(4)
# print("clicked", x)
plt.show()


x = np.array(x)
print(x)

# sift = cv2.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(src,None)
# kp2, des2 = sift.detectAndCompute(dst,None)
# bf = cv2.BFMatcher()
# matches = bf.match(des1,des2)
# # # im3 = cv2.drawMatches(src,kp1,dst,kp2,matches, flags = 2, outImg = None)
# # # plt.imshow(cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)), plt.show()

# src_pts = np.array([ kp1[m.queryIdx].pt for m in matches ])
# dst_pts = np.array([ kp2[m.trainIdx].pt for m in matches ])

# model, inliers = ransac((src_pts,
#                          dst_pts),
#                         AffineTransform, min_samples=3,
#                         residual_threshold=1)



# h = model.params



h, status = cv2.findHomography(x, c)



# print h.shape[0]



dst = cv2.warpPerspective(dst, h, (rows,cols))

plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# # # x = plt.ginput(4)
# # # print("clicked", x)
plt.show()

# sift = cv2.xfeatures2d.SIFT_create()
# src = cv2.imread('bookCover.jpg')
# kp1, des1 = sift.detectAndCompute(src,None)
# kp2, des2 = sift.detectAndCompute(dst,None)
# bf = cv2.BFMatcher()

# # # Match descriptors.
# matches = bf.match(des1,des2)
# # im3 = cv2.drawMatches(src,kp1,dst,kp2,matches, flags = 2, outImg = None)
# # plt.imshow(cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)), plt.show()

# src_pts = np.array([ kp1[m.queryIdx].pt for m in matches ])
# dst_pts = np.array([ kp2[m.trainIdx].pt for m in matches ])


# model, inliers = ransac((src_pts,
#                          dst_pts),
#                         AffineTransform, min_samples=3,
#                         residual_threshold=1)

# tform3 = tf.ProjectiveTransform()
# tform3.estimate(src, dst)
# warped = tf.warp(dst, tform3, output_shape=(50, 300))

# warped = warp(dst, model)
# plt.imshow(warped, cmap='gray'), plt.show()


# im3 = cv2.drawMatches(src,kp1,dst,kp2,matches, flags = 2, outImg = None)
# plt.imshow(cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)), plt.show()