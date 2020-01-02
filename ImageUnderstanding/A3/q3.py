import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import LineModelND, ransac
from skimage.transform import AffineTransform, warp
from skimage.feature import match_descriptors, ORB, plot_matches



img1 = cv2.imread('30cm.jpg')          # queryImage

# Initiate SIFT detector


h,w,ch= img1.shape

plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
x = np.array(plt.ginput(2))
plt.show()

print w, h
print x