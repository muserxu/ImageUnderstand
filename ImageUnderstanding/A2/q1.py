import numpy as np 
import random
import cv2 as cv2
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.filters as nd_filters
# from astropy.modeling.models import Gaussian2D
from scipy.ndimage import gaussian_filter
import math
import matplotlib.pyplot as plt


def imshowBGR2RGB( im ):
  img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.axis('off')
  return

def local_filter(x, order):
    x.sort()
    return x[order]
    
def ordfilt2(A, order, mask_size):
    return nd_filters.rank_filter(A, lambda x, ord=order: local_filter(x, ord), size=(mask_size, mask_size))


def nonMaximumSuppression(corner, radius, threshold):
    N = 50
    x_start = -1.0
    y_start= -1.0,

    x = np.linspace(x_start, radius*2, N)
    y = np.linspace(y_start, radius*2, N)

    x, y = np.meshgrid(x, y)

    r = np.sqrt((x - radius)**2 + (y - radius)**2)
    inside = (r <= radius)

    fig, ax = plt.subplots()
    ax.set(xlabel='X', ylabel='Y', aspect=1.0)
    ax.scatter(x[inside], y[inside])
    
    localMax = ordfilt2(corner.astype('float'), (radius**2), inside.astype('int'))
    index = np.where(localMax > threshold and corner == localMax)
    return index

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")



# In[ ]:

if __name__ == '__main__':
  img = cv2.imread('./images/building.jpg')

  imshowBGR2RGB(img)


  # In[ ]:


  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),7)
  Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
  Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)


  # In[ ]:


  plt.subplot(2,1,1), plt.imshow(Ix,cmap = 'gray')
  plt.subplot(2,1,2), plt.imshow(Iy,cmap = 'gray')


  # In[ ]:


  IxIy = np.multiply(Ix, Iy)
  Ix2 = np.multiply(Ix, Ix)
  Iy2 = np.multiply(Iy, Iy)


  # In[ ]:


  Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
  Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
  IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 


  # In[ ]:


  plt.subplot(1,3,1), plt.imshow(Ix2_blur,cmap = 'gray')
  plt.subplot(1,3,2), plt.imshow(Iy2_blur,cmap = 'gray')
  plt.subplot(1,3,3), plt.imshow(IxIy_blur,cmap = 'gray')


  # In[ ]:


  det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
  trace = Ix2_blur + Iy2_blur


  # In[ ]:


  plt.subplot(1,2,1), plt.imshow(det,cmap = 'gray')
  plt.subplot(1,2,2), plt.imshow(trace,cmap = 'gray')
  


  # In[ ]:
  np.seterr(divide='ignore', invalid='ignore')

# harris method
  R = det - 0.05 * np.multiply(trace,trace)
    
# brown's method
#   R = np.divide(det,trace)
#   plt.imshow(R, extent=[0, 1, 0, 1])

#   plt.subplot(1,2,1), plt.imshow(img), plt.axis('off')
#   plt.subplot(1,2,2), plt.imshow(R,cmap = 'gray'), plt.axis('off')
#   plt.show()
#   plt.savefig("Brown.png")


  #q1.b
  
  nonMaximumSuppression(R, 10 , 0.01)
#   plt.show()