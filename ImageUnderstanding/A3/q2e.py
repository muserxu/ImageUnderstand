import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import LineModelND, ransac
from skimage.transform import AffineTransform, warp
from skimage.feature import match_descriptors, ORB, plot_matches



img1 = cv2.imread('anotherBookCover.jpg', 0)          # queryImage
img2 = cv2.imread('im2.jpg', 0) # trainImage

# Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)


# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)

# flann = cv2.FlannBasedMatcher(index_params, search_params)

# matches = flann.knnMatch(des1,des2,k=2)

# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
# matchesMask = mask.ravel().tolist()

# h,w= img1.shape
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.perspectiveTransform(pts,M)

# img3 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
# # H, status = cv2.findHomography(dst, pts)
# # dst = cv2.warpPerspective(img2, H, (w,h))

# plt.imshow(dst, cmap="gray")
# plt.show()

# H, status = cv2.findHomography(x, pts)

# im_temp = cv2.warpPerspective(img2, H, (w,h))

# cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
    
#     # Add warped source image to destination image.
# im_dst = im_dst + im_temp

# plt.imshow(dst, cmap='gray')
# plt.show()


# im_src = cv2.imread('anotherBookCover.jpg')          # sorce
# im_dst = cv2.imread('im1.jpg') #dst

# h,w,ch= im_src.shape



# pts_src = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# plt.imshow(im_dst)
# pts_dst = np.array(plt.ginput(4))
# plt.show()

# H, status = cv2.findHomography(pts_src, pts_dst)

# h,w,ch= im_dst.shape

# im_temp = cv2.warpPerspective(im_src, H, (w,h))
# plt.imshow(im_temp), plt.show()
# print im_temp.shape[1], im_temp.shape[0]
# cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
# im_dst = im_dst + im_temp
# img3 = cv2.add(im_temp, img2[:w, :h])
# plt.imshow(cv2.cvtColor(im_dst, cv2.COLOR_BGR2RGB)), plt.show()
# # added_image = cv2.addWeighted(img2,0.4,im_temp,0.1,0)
# plt.imshow(dst),plt.show()


im_src = cv2.imread('anotherBookCover.jpg')
size = im_src.shape

# Create a vector of source points.
pts_src = np.array(
                    [
                    [0,0],
                    [size[1] - 1, 0],
                    [size[1] - 1, size[0] -1],

                    [0, size[0] - 1 ]
                    ],dtype=float
                    )


# Read destination image
im_dst = cv2.imread('im3.jpg')


plt.imshow(im_dst)
pts_dst = np.array(plt.ginput(4))
plt.show()

h, status = cv2.findHomography(pts_src, pts_dst)


im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)

im_dst = im_dst + im_temp
plt.imshow(cv2.cvtColor(im_dst, cv2.COLOR_BGR2RGB)), plt.show()
