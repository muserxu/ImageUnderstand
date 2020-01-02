import numpy as np 
import random
import cv2 as cv2
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.filters as nd_filters
# from astropy.modeling.models import Gaussian2D
from scipy.ndimage import gaussian_filter
import math
import matplotlib.pyplot as plt


im = cv2.imread('image1.jpg')
rows, cols, ch = im.shape
w = 210
h = 297

c = np.array([[0+500, 0+500], [w-1+500, 0+500], [w-1+500,h-1+500], [0+500,h-1+500]])

plt.imshow(im, origin='lower')

plt.xlim(500, 700)
plt.ylim(600, 200)
x = plt.ginput(4)
print("clicked", x)
plt.show()


x = np.array(x)
print(x)


h, status = cv2.findHomography(x, c)
print h


dst = cv2.warpPerspective(im, h, (rows,cols*4))
plt.imshow(dst)
x = plt.ginput(4)
print("clicked", x)
plt.show()


# output 
# C:\Users\muser\Desktop\420A3>python c:/Users/muser/Desktop/420A3/q1.py
# ('clicked', [(685.0973300712691, 311.33825221688204), (575.6312191159549, 303.2153252555336), (579.1124735279614, 499.7127965110094), (684.7105240254906, 496.6183481447814)])
# [[685.09733007 311.33825222]
#  [575.63121912 303.21532526]
#  [579.11247353 499.71279651]
#  [684.71052403 496.61834814]]
# 3
# ('clicked', [(197.31183819903072, 216.06187446209697), (932.552770014795, 182.64183210683495), (982.6828335476885, 2388.3646275541278), (218.1993646710689, 2321.5245428436037)])