from math import sqrt
from skimage import data
from skimage.feature import blob_log
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt


image = cv2.imread("./images/synthetic.png")
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, min_sigma = 1, max_sigma=40, num_sigma=25, threshold=.242)


blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

color = 'red'
title = 'Laplacian of Gaussian'


fig= plt.subplots(figsize=(9, 3), sharex=True, sharey=True)
fig[1].set_title("LoG")
fig[1].imshow(image, interpolation='nearest')



for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
    fig[1].add_patch(c)
fig[1].set_axis_off()

plt.tight_layout()
plt.show()
