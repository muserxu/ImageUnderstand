import cv2 
import numpy as np
import matplotlib.pyplot as plt

im1 = '004945'
im2 = '004964'
im3 = '005002'
results = 'data/test/results/'
left_disparity = '_left_disparity.png'
calib = 'data/test/calib'
allcalib = '_allcalib.txt'


im1_left = results + im1 + left_disparity
im2_left = results + im2 + left_disparity
im3_left = results + im3 + left_disparity

im1_calib = calib + im1 + allcalib
im2_calib = calib + im2 + allcalib
im3_calib = calib + im3 + allcalib


im1_dis = cv2.imread(im1_left, 0)
cols, rows= im1_dis.shape
im2_dis = cv2.imread(im2_left, 0)
im3_dis = cv2.imread(im3_left, 0)

im_dis = [im1_dis/256, im2_dis/256, im3_dis/256]

f = 721.537700
b = 0.5327119288 * 1000

fig = plt.figure()


for n in range(3):
    depth = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            depth[j,i] = (f-b)/im_dis[n][j,i]
            if (im_dis[n][j,i] == 0):
                depth[j,i] = 0
    fig.add_subplot(3,1, n+1)
    plt.imshow(depth)
plt.show()

