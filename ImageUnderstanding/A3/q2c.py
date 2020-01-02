import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import LineModelND, ransac
from skimage.transform import AffineTransform, warp
from skimage.feature import match_descriptors, ORB, plot_matches

im1 = cv2.imread('anotherBookCover.jpg',0)          # queryImage
im2 = cv2.imread('im1.jpg',0) # trainImage



sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(im1,None)
kp2, des2 = sift.detectAndCompute(im2,None)


bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.match(des1,des2)




src_pts = np.array([ kp1[m.queryIdx].pt for m in matches ])
dst_pts = np.array([ kp2[m.trainIdx].pt for m in matches ])




model, inliers = ransac((src_pts,
                         dst_pts),
                        AffineTransform, min_samples=3,
                        residual_threshold=1)


# final_matches = []
# for i in range(len(inliers)):
#     if inliers[i] == True:
#         final_matches.append(matches[i])

# src_pts = np.array([ kp1[m.queryIdx].pt for m in final_matches ])
# dst_pts = np.array([ kp2[m.trainIdx].pt for m in final_matches ])
# img3 = cv2.polylines(im2,[np.int32(inliers)],True,255,3, cv2.LINE_AA)

warped = warp(im2, model)

plt.imshow(warped, cmap='gray'), plt.show()



# dst = cv2.warpAffine(im1,M,(cols,rows))
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()





# im3 = cv2.drawMatches(im1,kp1,im2,kp2,final_matches, flags = 2, outImg = None)
# # im3 = cv2.drawMatches(im1,kp1,im2,kp2,matches, flags = 2, outImg = None)
# plt.imshow(im3),plt.show()

# import numpy as np
# from skimage import data
# from skimage.color import rgb2gray
# from skimage.feature import match_descriptors, ORB, plot_matches
# from skimage.measure import ransac
# from skimage.transform import AffineTransform
# import matplotlib.pyplot as plt
# import cv2

# np.random.seed(0)


# img_left = cv2.imread('bookCover.jpg',0)          # queryImage
# img_right = cv2.imread('im1.jpg',0) # trainImage


# # Find sparse feature correspondences between left and right image.

# descriptor_extractor = ORB()

# descriptor_extractor.detect_and_extract(img_left)
# keypoints_left = descriptor_extractor.keypoints
# descriptors_left = descriptor_extractor.descriptors

# descriptor_extractor.detect_and_extract(img_right)
# keypoints_right = descriptor_extractor.keypoints
# descriptors_right = descriptor_extractor.descriptors

# matches = match_descriptors(descriptors_left, descriptors_right,
#                             cross_check=True)


# # Estimate the epipolar geometry between the left and right image.

# model, inliers = ransac((keypoints_left[matches[:, 0]],
#                          keypoints_right[matches[:, 1]]),
#                         AffineTransform, min_samples=3,
#                         residual_threshold=1, max_trials=5000)

# print(model.params)


# inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
# inlier_keypoints_right = keypoints_right[matches[inliers, 1]]


# fig, ax = plt.subplots(nrows=1, ncols=1)
# plt.gray()

# plot_matches(ax,img_left, img_right, keypoints_left, keypoints_right,
#              matches[inliers], only_matches=True)

# plt.show()






